"""당도 회귀 모델 학습.

    python train.py --augments geometric --tag geo
    python train.py --split session --test-sessions 1440x1081 1080x1440

``predictor.py``의 ``BrixRegression6CH_Deep``을 그대로 학습하므로, 결과 가중치를
``brix_model/``에 두면 추론 코드가 바로 씁니다.

측정 원칙
---------
* 검증 셋으로 조기 종료 지점을 고르고, **테스트 셋은 마지막에 한 번만** 봅니다.
* 증강은 학습 셋에만 겁니다.
* 비교 기준선은 "항상 학습 셋 평균만 답하는 모델"입니다. 이걸 못 이기면
  모델이 이미지에서 아무것도 못 배운 것입니다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dataset import (COLOR_SPACE_CHANNELS, BrixCropDataset, session_of,  # noqa: E402
                     session_split, stratified_split)
from predictor import BrixRegression6CH_Deep  # noqa: E402


def metrics(pred, true):
    err = pred - true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "bias": float(err.mean()),
        "pred_std": float(pred.std()),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_once(args, train_idx, val_idx, test_idx, df, verbose=True):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    color_space = getattr(args, "color_space", "hsv")

    def make(idx, augments, shuffle):
        ds = BrixCropDataset(df.images[idx], df.brix[idx], args.crops,
                             augments=augments, seed=args.seed, color_space=color_space)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make(train_idx, args.augments, True)
    val_loader = make(val_idx, (), False)
    test_loader = make(test_idx, (), False)

    model = BrixRegression6CH_Deep(COLOR_SPACE_CHANNELS[color_space]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 라벨에 이상치가 섞여 있어도 덜 흔들리도록 Huber를 씁니다.
    criterion = nn.SmoothL1Loss(beta=1.0)

    best = {"val_mae": float("inf"), "epoch": -1, "state": None}
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
        scheduler.step()

        vp, vt = evaluate(model, val_loader, device)
        vm = metrics(vp, vt)
        history.append({"epoch": epoch, "train_loss": total / len(train_idx), **vm})

        if vm["mae"] < best["val_mae"]:
            best = {"val_mae": vm["mae"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        if verbose and (epoch % args.log_every == 0 or epoch == 1):
            print("  epoch {:>3}  train_loss {:.4f}  val_mae {:.4f}  (best {:.4f} @ {})".format(
                epoch, total / len(train_idx), vm["mae"], best["val_mae"], best["epoch"]))

        if args.patience and epoch - best["epoch"] >= args.patience:
            if verbose:
                print("  조기 종료: {} epoch 동안 검증 개선 없음".format(args.patience))
            break

    # 마지막 epoch 모델의 성능도 따로 잽니다.
    # 검증 셋이 48장뿐이라 "best epoch 고르기" 자체가 운에 좌우됩니다.
    # 고정 예산으로 끝까지 학습한 결과를 함께 봐야 증강 효과를 공정하게 비교할 수 있습니다.
    fp, ft = evaluate(model, test_loader, device)
    final_metrics = metrics(fp, ft)

    model.load_state_dict(best["state"])
    tp, tt = evaluate(model, test_loader, device)
    result = {
        "tag": args.tag,
        "augments": list(args.augments),
        "color_space": color_space,
        "split": args.split,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "best_epoch": best["epoch"],
        "val_mae": best["val_mae"],
        "test": metrics(tp, tt),
        "test_final_epoch": final_metrics,
        "last_epoch": len(history),
        "baseline_test_mae": float(np.abs(tt - df.brix[train_idx].mean()).mean()),
        "seconds": time.time() - t0,
        # 사후 분석(등급 임계값 재보정 등)을 위해 테스트 예측을 남깁니다.
        "test_images": [str(v) for v in df.images[test_idx]],
        "test_pred": [float(v) for v in tp],
        "test_true": [float(v) for v in tt],
        "train_brix_mean": float(df.brix[train_idx].mean()),
    }
    return model, result, history, (tp, tt)


def main():
    p = argparse.ArgumentParser(description="당도 회귀 모델 학습")
    p.add_argument("--crops", default=str(config.PROJECT_ROOT / "data" / "crops"))
    p.add_argument("--csv", default=str(config.PROJECT_ROOT / "data" / "crops.csv"))
    p.add_argument("--out", default=str(config.PROJECT_ROOT / "data" / "runs"))
    p.add_argument("--tag", default="run")
    p.add_argument("--augments", nargs="*", default=[],
                   choices=["geometric", "brightness", "color"])
    p.add_argument("--split", choices=["stratified", "session"], default="stratified")
    p.add_argument("--color-space", choices=list(COLOR_SPACE_CHANNELS), default="hsv",
                   help="hsv=현재 방식, huesincos=색조 끊김을 없앤 7채널")
    p.add_argument("--test-sessions", nargs="*", default=["1440x1081", "1080x1440"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-weights", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.split == "session":
        tr, va, te = session_split(df, args.test_sessions, seed=args.seed)
    else:
        tr, va, te = stratified_split(df, seed=args.seed)

    print("분할: 학습 {} / 검증 {} / 테스트 {}  ({})".format(len(tr), len(va), len(te), args.split))
    print("  학습 Brix 평균 {:.2f}, 테스트 Brix 평균 {:.2f}".format(
        df.brix[tr].mean(), df.brix[te].mean()))
    print("증강: {}".format(args.augments or "없음"))

    model, result, history, _ = train_once(args, tr, va, te, df)

    print("\n테스트 결과 (best epoch {})".format(result["best_epoch"]))
    t = result["test"]
    print("  MAE  {:.3f}   (기준선 '항상 평균' {:.3f})".format(t["mae"], result["baseline_test_mae"]))
    print("  RMSE {:.3f}   R² {:.3f}   편향 {:+.3f}   예측 표준편차 {:.3f}".format(
        t["rmse"], t["r2"], t["bias"], t["pred_std"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"result": result, "history": history, "args": vars(args)},
              open(out / "{}.json".format(args.tag), "w"), indent=2)
    if args.save_weights:
        torch.save(model.state_dict(), out / "{}.pth".format(args.tag))
    print("기록: {}".format(out / "{}.json".format(args.tag)))


if __name__ == "__main__":
    main()
