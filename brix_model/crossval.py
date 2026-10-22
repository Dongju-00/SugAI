"""k-fold 교차검증으로 조합을 비교합니다.

    python crossval.py                      # 무작위 5-fold
    python crossval.py --split session      # 세션 통째로 빼는 k-fold

왜 k-fold인가
-------------
단일 분할(`train.py`)은 테스트 50장 한 벌만 봅니다. 조합 간 차이가 0.1 Brix 수준인데
시드만 바꿔도 0.08쯤 흔들려서, 순위가 노이즈에 묻힙니다. k-fold는 247장 전부가 한 번씩
테스트가 되므로 추정이 훨씬 안정적입니다.

두 분할의 의미가 다릅니다.
  random  : 같은 촬영 조건의 새 딸기에서 얼마나 하는가
  session : **처음 보는 촬영 조건**의 새 딸기에서 얼마나 하는가 (훨씬 혹독)

이 데이터는 촬영 세션이 당도와 엮여 있어(해상도만으로 Brix를 R² 0.408로 예측 가능)
두 숫자가 크게 벌어질 수 있습니다. 벌어진다면 모델이 딸기가 아니라 촬영 조건을
외우고 있다는 뜻입니다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import kfold_splits, session_kfold_splits  # noqa: E402
from grades import CURRENT_THRESHOLDS, calibrate_thresholds, grade_metrics  # noqa: E402
from train import train_once  # noqa: E402

#: 기존 가중치. fine-tuning 출발점으로 씁니다.
_W = str(config.BRIX_MODEL_DIR / "best_model(Regressor)_fold1.pth")

#: (표시 이름, 증강 목록, 색 표현, fine-tuning 출발점 or None)
CONFIGS = [
    ("증강 없음",                  [],                                   "hsv",       None),
    ("기하 (뒤집기·회전)",          ["geometric"],                        "hsv",       None),
    ("밝기 ±20%",                 ["brightness"],                       "hsv",       None),
    ("기하 + 밝기",                ["geometric", "brightness"],          "hsv",       None),
    ("기하 + 밝기 + 색",           ["geometric", "brightness", "color"], "hsv",       None),
    ("기하 + 밝기 · Hue sin/cos",  ["geometric", "brightness"],          "huesincos", None),
    ("증강 없음 · fine-tune",      [],                                   "hsv",       _W),
    ("기하 + 밝기 · fine-tune",    ["geometric", "brightness"],          "hsv",       _W),
]


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def pooled_metrics(pred, true, train_mean):
    err = pred - true
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot else float("nan"),
        "pred_std": float(pred.std()),
        "bias": float(err.mean()),
        "baseline_mae": float(np.abs(true - train_mean).mean()),
    }


def main():
    p = argparse.ArgumentParser(description="k-fold 교차검증으로 조합 비교")
    p.add_argument("--crops", default=str(config.PROJECT_ROOT / "data" / "crops"))
    p.add_argument("--csv", default=str(config.PROJECT_ROOT / "data" / "crops.csv"))
    p.add_argument("--out", default=str(config.PROJECT_ROOT / "data" / "runs"))
    p.add_argument("--split", choices=["random", "session"], default="random")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=0, help="0이면 조기 종료 없음")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu. 생략하면 자동")
    cli = p.parse_args()

    df = pd.read_csv(cli.csv)
    if cli.split == "session":
        raw = session_kfold_splits(df, seed=cli.seed)
        splits = [(tr, va, te) for tr, va, te, _ in raw]
        names = [n for *_, n in raw]
    else:
        splits = kfold_splits(df, k=cli.k, seed=cli.seed)
        names = ["fold{}".format(i + 1) for i in range(len(splits))]

    print("{} {}-fold  (테스트 총 {}장)".format(cli.split, len(splits), len(df)), flush=True)
    for n, (tr, va, te) in zip(names, splits):
        print("  {:<10} 학습 {:>3} 검증 {:>2} 테스트 {:>3}".format(n, len(tr), len(va), len(te)),
              flush=True)

    results = []
    for label, augs, cspace, init_from in CONFIGS:
        preds, trues, train_means, per_fold = [], [], [], []
        print("\n[{}]  증강 {}, 색표현 {}".format(label, augs or "없음", cspace), flush=True)
        for n, (tr, va, te) in zip(names, splits):
            args = Args(crops=cli.crops,
                        tag="{}_cv_{}_{}_{}".format(cli.split, "+".join(augs) or "none", cspace, n),
                        augments=augs, split=cli.split, epochs=cli.epochs,
                        patience=cli.patience, batch_size=32, lr=1e-3, weight_decay=1e-4,
                        seed=cli.seed, log_every=10 ** 9, color_space=cspace,
                        init_from=init_from, device=cli.device)
            _, res, _, _ = train_once(args, tr, va, te, df, verbose=False)
            preds.append(np.array(res["test_pred"]))
            trues.append(np.array(res["test_true"]))
            train_means.append(res["train_brix_mean"])
            per_fold.append({"fold": n, "device": res.get("device"), **res["test"]})
            print("    {:<10} MAE {:.3f}  R2 {:+.3f}  ({:.0f}s)".format(
                n, res["test"]["mae"], res["test"]["r2"], res["seconds"]), flush=True)

        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        # fold마다 학습 평균이 다르므로 기준선도 fold별로 계산해 이어 붙입니다.
        base = np.concatenate([np.abs(t - m) for t, m in zip(trues, train_means)])
        m = pooled_metrics(pred, true, true.mean())
        m["baseline_mae"] = float(base.mean())
        g_now = grade_metrics(pred, true, CURRENT_THRESHOLDS)
        fitted = calibrate_thresholds(pred, true)
        g_fit = grade_metrics(pred, true, fitted, true_thresholds=CURRENT_THRESHOLDS)

        results.append({"label": label, "augments": augs, "color_space": cspace,
                        "fine_tuned": init_from is not None,
                        "split": cli.split, "pooled": m, "per_fold": per_fold,
                        "grade_current": g_now, "grade_calibrated": g_fit,
                        "pred": pred.tolist(), "true": true.tolist()})
        print("    => 전체 {}장: MAE {:.3f}  R2 {:+.3f}  예측편차 {:.2f}  (기준선 {:.3f})".format(
            len(pred), m["mae"], m["r2"], m["pred_std"], m["baseline_mae"]), flush=True)
        json.dump(results, open(Path(cli.out) / "crossval_{}.json".format(cli.split), "w"), indent=2)

    print("\n" + "=" * 86)
    print("{} {}-fold 종합 (테스트 {}장 전체)".format(cli.split, len(splits), len(df)))
    print("=" * 86)
    print("epoch {} · 장치 {}".format(cli.epochs, results[0]["per_fold"][0].get("device", "?")))
    print("{:<28} {:>8} {:>8} {:>9} {:>9} {:>9}".format(
        "조합", "MAE", "R²", "예측편차", "등급정확도", "C재현율"))
    for r in results:
        m, g = r["pooled"], r["grade_current"]
        print("{:<28} {:>8.3f} {:>8.3f} {:>9.2f} {:>8.1%} {:>9.0%}".format(
            r["label"], m["mae"], m["r2"], m["pred_std"], g["accuracy"], g["recall"]["C"]))
    print("{:<28} {:>8.3f} {:>8.3f}".format("기준선(항상 평균)", results[0]["pooled"]["baseline_mae"], 0.0))


if __name__ == "__main__":
    main()
