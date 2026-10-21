"""증강·분할 조합을 한 번에 돌려 held-out 성능을 비교합니다.

    python experiments.py                 # 전체
    python experiments.py --seeds 0 1 2   # 시드를 늘려 안정성 확인

각 조합을 같은 분할·같은 시드로 돌리므로, 차이가 나면 그건 조합의 효과입니다.
결과는 ``data/runs/experiments.json``에 쌓입니다.
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
from dataset import session_split, stratified_split  # noqa: E402
from train import train_once  # noqa: E402


#: (표시 이름, 증강 목록, 색 표현)
CONFIGS = [
    ("증강 없음",                   [],                                   "hsv"),
    ("기하 (뒤집기·회전)",           ["geometric"],                        "hsv"),
    ("밝기 ±20%",                  ["brightness"],                       "hsv"),
    ("색조·채도 (±6도/±20%)",       ["color"],                            "hsv"),
    ("기하 + 밝기",                 ["geometric", "brightness"],          "hsv"),
    ("기하 + 밝기 + 색",            ["geometric", "brightness", "color"], "hsv"),
    ("기하 + 밝기 · Hue sin/cos",   ["geometric", "brightness"],          "huesincos"),
]


class Args:
    """train_once가 기대하는 속성 묶음."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops", default=str(config.PROJECT_ROOT / "data" / "crops"))
    p.add_argument("--csv", default=str(config.PROJECT_ROOT / "data" / "crops.csv"))
    p.add_argument("--out", default=str(config.PROJECT_ROOT / "data" / "runs"))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    p.add_argument("--split", choices=["stratified", "session"], default="stratified")
    p.add_argument("--test-sessions", nargs="*", default=["1440x1081", "1080x1440"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    cli = p.parse_args()

    df = pd.read_csv(cli.csv)
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in cli.seeds:
        if cli.split == "session":
            tr, va, te = session_split(df, cli.test_sessions, seed=seed)
        else:
            tr, va, te = stratified_split(df, seed=seed)
        baseline = float(np.abs(df.brix[te] - df.brix[tr].mean()).mean())
        print("\n{} seed={} 분할: 학습 {} / 검증 {} / 테스트 {}  기준선 MAE {:.3f}".format(
            cli.split, seed, len(tr), len(va), len(te), baseline), flush=True)

        for label, augs, cspace in CONFIGS:
            args = Args(crops=cli.crops,
                        tag="{}_{}_{}_s{}".format(cli.split, "+".join(augs) or "none", cspace, seed),
                        augments=augs, split=cli.split, epochs=cli.epochs, patience=cli.patience,
                        batch_size=32, lr=1e-3, weight_decay=1e-4, seed=seed, log_every=10**9,
                        color_space=cspace)
            _, res, _, _ = train_once(args, tr, va, te, df, verbose=False)
            res["label"] = label
            res["seed"] = seed
            results.append(res)
            t, f = res["test"], res["test_final_epoch"]
            print("  {:<26} bestep MAE {:.3f} (R2 {:+.3f})   최종ep MAE {:.3f} (R2 {:+.3f})"
                  "   pred_std {:.2f}  [{:.0f}s, best ep{}/{}]".format(
                      label, t["mae"], t["r2"], f["mae"], f["r2"], t["pred_std"],
                      res["seconds"], res["best_epoch"], res["last_epoch"]), flush=True)
            json.dump(results, open(out / "experiments_{}.json".format(cli.split), "w"), indent=2)

    print("\n" + "=" * 74)
    print("시드 평균 ({} 분할, seeds={})".format(cli.split, cli.seeds))
    print("=" * 74)
    print("{:<28} {:>9} {:>9} {:>8} {:>9}".format("조합", "MAE(best)", "MAE(최종)", "R²", "예측편차"))
    for label, _, _ in CONFIGS:
        rs = [r for r in results if r["label"] == label]
        if not rs:
            continue
        print("{:<28} {:>9.3f} {:>9.3f} {:>8.3f} {:>9.2f}".format(
            label,
            np.mean([r["test"]["mae"] for r in rs]),
            np.mean([r["test_final_epoch"]["mae"] for r in rs]),
            np.mean([r["test"]["r2"] for r in rs]),
            np.mean([r["test"]["pred_std"] for r in rs])))
    print("{:<28} {:>9.3f}  <- 이미지를 전혀 안 보는 기준선".format(
        "항상 학습셋 평균", np.mean([r["baseline_test_mae"] for r in results])))


if __name__ == "__main__":
    main()
