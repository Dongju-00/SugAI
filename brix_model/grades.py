"""당도 예측값을 S/A/B/C 등급으로 바꾸는 규칙과 그 평가.

현재 운영 임계값 ``[5.56, 6.97, 9.23]``에는 구조적 문제가 있습니다. 기존 모델의
예측 최소값이 5.606이라 C 조건(``< 5.56``)에 **한 번도 닿지 못합니다.** 실측이 C인
33장을 전부 놓치고, 선별기는 C 칸으로 한 번도 동작하지 않습니다.

이 모듈은 그 상태를 눈으로 확인하고(``grade_report``), 예측 분포에 맞게 임계값을
다시 잡는(``calibrate_thresholds``) 도구입니다.

    python grades.py ../data/runs/stratified_none_hsv_s0.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

#: 현재 운영 중인 임계값. (C|B 경계, B|A 경계, A|S 경계)
CURRENT_THRESHOLDS = (5.56, 6.97, 9.23)
#: 구버전 legacy/brix_model/model.py가 쓰던 값. 비교용.
LEGACY_THRESHOLDS = (6.05, 6.85, 7.75)

GRADES = ("S", "A", "B", "C")


def to_grade(brix, thresholds=CURRENT_THRESHOLDS):
    """당도 값(또는 배열)을 등급 문자로 바꿉니다."""
    low, mid, high = thresholds
    v = np.asarray(brix, dtype=float)
    out = np.where(v < low, "C", np.where(v < mid, "B", np.where(v < high, "A", "S")))
    return out if out.ndim else str(out)


def confusion(pred_grade, true_grade):
    """``[실측][예측]`` 순서의 4x4 혼동행렬 (행·열 모두 S, A, B, C 순)."""
    return np.array([[int(((true_grade == t) & (pred_grade == p)).sum())
                      for p in GRADES] for t in GRADES])


def grade_metrics(pred, true, thresholds=CURRENT_THRESHOLDS, true_thresholds=None):
    """예측 당도를 등급화해 실측 등급과 비교합니다.

    ``true_thresholds``는 **실측 당도를 등급으로 바꾸는 기준**이고, 프로젝트가 정한
    등급 정의입니다. 예측 쪽 임계값(``thresholds``)을 재보정해 볼 때도 이건 고정해야
    합니다. 둘을 같이 움직이면 정답지가 따라 바뀌어 비교가 성립하지 않습니다.
    생략하면 같은 값을 써서 "같은 규칙끼리" 비교합니다.
    """
    if true_thresholds is None:
        true_thresholds = thresholds
    gp, gt = to_grade(pred, thresholds), to_grade(true, true_thresholds)
    cm = confusion(gp, gt)
    recall = {}
    for i, g in enumerate(GRADES):
        n = cm[i].sum()
        recall[g] = float(cm[i, i] / n) if n else float("nan")
    return {
        "thresholds": list(thresholds),
        "true_thresholds": list(true_thresholds),
        "accuracy": float((gp == gt).mean()),
        "recall": recall,
        "confusion": cm.tolist(),
        "pred_counts": {g: int((gp == g).sum()) for g in GRADES},
        "true_counts": {g: int((gt == g).sum()) for g in GRADES},
        "unreachable": [g for g in GRADES if (gp == g).sum() == 0],
    }


def calibrate_thresholds(pred, true, true_thresholds=CURRENT_THRESHOLDS):
    """예측 등급 분포가 실측 등급 분포와 같아지도록 **예측 쪽** 임계값만 다시 잡습니다.

    모델 예측이 가운데로 수축되어 있으면 원래 임계값에서는 바깥 등급이 아예 나오지
    않습니다. 실측 등급 비율(프로젝트가 정한 기준)은 그대로 두고, 그 비율에 해당하는
    **예측값의 분위수**를 새 경계로 씁니다.

    주의: 테스트 셋에서 직접 뽑으면 그 셋에 맞춘 값이 됩니다. 실제로는 학습/검증 셋의
    예측으로 구해 테스트에 적용해야 합니다.
    """
    gt = to_grade(true, true_thresholds)
    # C, B, A 까지의 누적 비율 -> 예측값의 같은 분위수
    cum = np.cumsum([(gt == g).mean() for g in ("C", "B", "A")])
    return tuple(float(v) for v in np.quantile(np.asarray(pred, float), cum))


def grade_report(pred, true, thresholds=CURRENT_THRESHOLDS, title="", true_thresholds=None):
    """등급 성능을 사람이 읽을 수 있게 출력합니다."""
    m = grade_metrics(pred, true, thresholds, true_thresholds)
    low, mid, high = thresholds
    print("{}  예측 임계값 C<{:.2f}<=B<{:.2f}<=A<{:.2f}<=S".format(title, low, mid, high))
    print("  등급 정확도 {:.1%}   (n={})".format(m["accuracy"], len(np.asarray(pred))))
    print("  실측 분포 {}".format(m["true_counts"]))
    print("  예측 분포 {}".format(m["pred_counts"]))
    if m["unreachable"]:
        print("  [경고] 한 번도 나오지 않은 등급: {} — 선별기가 그 칸으로 동작하지 않습니다"
              .format(", ".join(m["unreachable"])))
    print("  등급별 재현율 " + "  ".join(
        "{} {:.0%}".format(g, m["recall"][g]) if m["recall"][g] == m["recall"][g] else "{} -".format(g)
        for g in GRADES))
    print("  혼동행렬 (행=실측, 열=예측)")
    print("        " + "".join("{:>5}".format(g) for g in GRADES))
    for g, row in zip(GRADES, m["confusion"]):
        print("     {} ".format(g) + "".join("{:>5}".format(v) for v in row))
    return m


def main():
    p = argparse.ArgumentParser(description="학습 결과의 등급 성능 분석과 임계값 재보정")
    p.add_argument("run_json", help="train.py 또는 experiments.py가 남긴 JSON")
    args = p.parse_args()

    data = json.load(open(args.run_json))
    runs = data if isinstance(data, list) else [data["result"]]

    for run in runs:
        if "test_pred" not in run:
            print("[건너뜀] {}: 예측값이 없습니다 (train.py 업데이트 이전 기록)"
                  .format(run.get("tag", "?")))
            continue
        pred = np.array(run["test_pred"])
        true = np.array(run["test_true"])
        label = run.get("label") or run.get("tag", "")
        print("\n" + "=" * 70)
        print("{}  (증강 {}, 색표현 {})".format(
            label, run.get("augments") or "없음", run.get("color_space", "hsv")))
        print("=" * 70)
        grade_report(pred, true, CURRENT_THRESHOLDS, "[현재 임계값]")
        print()
        fitted = calibrate_thresholds(pred, true)
        # 정답지(실측 등급)는 프로젝트 기준으로 고정하고 예측 쪽만 보정합니다.
        grade_report(pred, true, fitted, "[예측 분포에 맞춘 임계값]",
                     true_thresholds=CURRENT_THRESHOLDS)


if __name__ == "__main__":
    main()
