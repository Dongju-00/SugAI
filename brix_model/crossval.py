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


def _cfg(label, augments=(), color_space="hsv", init_from=None, width=1.0, blocks=4):
    return {"label": label, "augments": list(augments), "color_space": color_space,
            "init_from": init_from, "width": width, "blocks": blocks}


#: --preset 으로 고릅니다.
PRESETS = {
    # 증강·색표현·fine-tuning 비교
    "augment": [
        _cfg("증강 없음"),
        _cfg("기하 (뒤집기·회전)", ["geometric"]),
        _cfg("밝기 ±20%", ["brightness"]),
        _cfg("기하 + 밝기", ["geometric", "brightness"]),
        _cfg("기하 + 밝기 + 색", ["geometric", "brightness", "color"]),
        _cfg("기하 + 밝기 · Hue sin/cos", ["geometric", "brightness"], "huesincos"),
        _cfg("증강 없음 · fine-tune", init_from=_W),
        _cfg("기하 + 밝기 · fine-tune", ["geometric", "brightness"], init_from=_W),
    ],
    # 모델 용량만 바꿔가며 비교. 파라미터의 73%가 layer4에 있어 blocks=3이 크게 줄입니다.
    "capacity": [
        _cfg("1.21M · 32/64/128/256 (현재)", width=1.0, blocks=4),
        _cfg("303K · width 0.5", width=0.5, blocks=4),
        _cfg("305K · blocks 3 (layer4 제거)", width=1.0, blocks=3),
        _cfg("77K · width 0.5 + blocks 3", width=0.5, blocks=3),
        _cfg("20K · width 0.25 + blocks 3", width=0.25, blocks=3),
        _cfg("5K · width 0.25 + blocks 2", width=0.25, blocks=2),
    ],
    # 같은 용량 스윕을 기하+밝기 증강과 함께 (용량 x 증강 상호작용 확인)
    "capacity_aug": [
        _cfg("1.21M + 증강", ["geometric", "brightness"], width=1.0, blocks=4),
        _cfg("303K + 증강", ["geometric", "brightness"], width=0.5, blocks=4),
        _cfg("305K + 증강", ["geometric", "brightness"], width=1.0, blocks=3),
        _cfg("77K + 증강", ["geometric", "brightness"], width=0.5, blocks=3),
        _cfg("20K + 증강", ["geometric", "brightness"], width=0.25, blocks=3),
        _cfg("5K + 증강", ["geometric", "brightness"], width=0.25, blocks=2),
    ],
}


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
    p.add_argument("--preset", choices=sorted(PRESETS), default="augment")
    p.add_argument("--tag-suffix", default="", help="결과 파일명에 붙일 꼬리표")
    cli = p.parse_args()
    configs = PRESETS[cli.preset]

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
    for c in configs:
        label, augs, cspace = c["label"], c["augments"], c["color_space"]
        init_from, width, blocks = c["init_from"], c["width"], c["blocks"]
        preds, trues, train_means, per_fold = [], [], [], []
        print("\n[{}]  증강 {}, 색표현 {}, width {} blocks {}".format(
            label, augs or "없음", cspace, width, blocks), flush=True)
        for n, (tr, va, te) in zip(names, splits):
            args = Args(crops=cli.crops,
                        tag="{}_cv_{}_{}_{}".format(cli.split, "+".join(augs) or "none", cspace, n),
                        augments=augs, split=cli.split, epochs=cli.epochs,
                        patience=cli.patience, batch_size=32, lr=1e-3, weight_decay=1e-4,
                        seed=cli.seed, log_every=10 ** 9, color_space=cspace,
                        init_from=init_from, device=cli.device,
                        width=width, blocks=blocks)
            _, res, _, _ = train_once(args, tr, va, te, df, verbose=False)
            preds.append(np.array(res["test_pred"]))
            trues.append(np.array(res["test_true"]))
            train_means.append(res["train_brix_mean"])
            per_fold.append({"fold": n, "device": res.get("device"),
                             "n_params": res.get("n_params"), **res["test"]})
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
                        "width": width, "blocks": blocks,
                        "n_params": per_fold[0].get("n_params"),
                        "split": cli.split, "pooled": m, "per_fold": per_fold,
                        "grade_current": g_now, "grade_calibrated": g_fit,
                        "pred": pred.tolist(), "true": true.tolist()})
        print("    => 전체 {}장: MAE {:.3f}  R2 {:+.3f}  예측편차 {:.2f}  (기준선 {:.3f})".format(
            len(pred), m["mae"], m["r2"], m["pred_std"], m["baseline_mae"]), flush=True)
        json.dump(results, open(Path(cli.out) / "crossval_{}{}.json".format(
            cli.split, cli.tag_suffix), "w"), indent=2)

    print("\n" + "=" * 86)
    print("{} {}-fold 종합 (테스트 {}장 전체)".format(cli.split, len(splits), len(df)))
    print("=" * 86)
    print("epoch {} · 장치 {}".format(cli.epochs, results[0]["per_fold"][0].get("device", "?")))
    print("{:<30} {:>10} {:>8} {:>8} {:>9}".format(
        "조합", "파라미터", "MAE", "R²", "예측편차"))
    for r in results:
        m = r["pooled"]
        print("{:<30} {:>10,} {:>8.3f} {:>8.3f} {:>9.2f}".format(
            r["label"], r.get("n_params") or 0, m["mae"], m["r2"], m["pred_std"]))
    print("{:<28} {:>8.3f} {:>8.3f}".format("기준선(항상 평균)", results[0]["pooled"]["baseline_mae"], 0.0))


if __name__ == "__main__":
    main()
