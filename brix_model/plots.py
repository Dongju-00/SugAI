"""측정 결과를 그림으로 만듭니다.

    python plots.py                    # 전체
    python plots.py --only 3 5         # 특정 그림만

결과는 ``data/figures/``에 저장됩니다. 숫자 표만으로는 놓치기 쉬운 것들
(예측 포화, 배치 간/내 신호 분리, Hue 끊김)을 눈으로 확인하는 용도입니다.

입력은 저장소 안에서 모두 구합니다.
  - ``data/runs/crossval_*.json``  교차검증 결과
  - ``data/crops/``                딸기 crop 이미지
  - ``train_images/``, ``labels.csv``
  - ``brix_model/*.pth``           기존(전체 프레임 학습) 모델
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# 한글 라벨이 깨지지 않도록 설치된 폰트 중 하나를 고릅니다.
for _f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트는 유니코드 마이너스가 깨집니다
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

OUT = config.PROJECT_ROOT / "data" / "figures"
RUNS = config.PROJECT_ROOT / "data" / "runs"
C_OLD, C_NEW, C_WARN = "#9aa0a6", "#1a73e8", "#d93025"


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print("  저장: {}".format(path))


def _mask(bgr):
    """HSV 빨강 마스크로 딸기 영역만 남깁니다 (가장 큰 덩어리)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)) | \
        cv2.inRange(hsv, (168, 70, 50), (179, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return m if n <= 1 else (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255


# ===============================
# 1) 문제 — 딸기가 프레임에서 차지하는 비율
# ===============================
def fig_area_fraction():
    labels = pd.read_csv(config.PROJECT_ROOT / "labels.csv")
    fr = []
    for name in labels.images:
        bgr = cv2.imread(str(config.PROJECT_ROOT / "train_images" / name))
        small = cv2.resize(bgr, (384, 384), interpolation=cv2.INTER_AREA)
        fr.append((_mask(small) > 0).mean())
    fr = np.array(fr) * 100

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # 학습 데이터가 0~3%에 몰려 있어 63%까지 한 축에 그리면 히스토그램이 안 보입니다.
    # 데이터 범위로 축을 잡고, 운영 입력 위치는 화살표로 가리킵니다.
    ax[0].hist(fr, bins=30, color=C_OLD, edgecolor="white")
    ax[0].axvline(fr.mean(), color=C_WARN, ls="--", lw=2,
                  label="학습 데이터 평균 {:.2f}%".format(fr.mean()))
    ax[0].set_xlim(0, max(fr.max() * 1.05, 3.2))
    ax[0].annotate("운영 입력 ≈ 63%\n(축 밖)", xy=(ax[0].get_xlim()[1], ax[0].get_ylim()[1] * 0.55),
                   xytext=(ax[0].get_xlim()[1] * 0.52, ax[0].get_ylim()[1] * 0.75),
                   fontsize=9, color=C_NEW, ha="center",
                   arrowprops=dict(arrowstyle="->", color=C_NEW, lw=1.8))
    ax[0].set_xlabel("딸기가 프레임에서 차지하는 면적 (%)")
    ax[0].set_ylabel("이미지 수")
    ax[0].set_title("학습 이미지는 딸기가 1% 남짓 (최대 {:.1f}%)".format(fr.max()))
    ax[0].legend(loc="upper right", fontsize=9)

    px = fr / 100 * 128 * 128
    ax[1].bar(["학습 데이터\n(전체 사진)", "운영 입력\n(YOLO crop)"],
              [px.mean(), 0.63 * 128 * 128], color=[C_OLD, C_NEW])
    ax[1].set_ylabel("128×128 안에서 딸기 픽셀 수")
    ax[1].set_title("같은 128×128인데 딸기 해상도가 {:.0f}배 차이".format(
        0.63 * 128 * 128 / px.mean()))
    for i, v in enumerate([px.mean(), 0.63 * 128 * 128]):
        ax[1].text(i, v, "{:.0f}px\n(≈{:.0f}×{:.0f})".format(v, np.sqrt(v), np.sqrt(v)),
                   ha="center", va="bottom")
    ax[1].set_ylim(0, 12500)
    fig.suptitle("[문제 1] 학습 이미지와 운영 입력의 형태가 다르다", fontsize=13, y=1.02)
    _save(fig, "1_area_fraction.png")


# ===============================
# 2) 문제 — 면적비에 따른 예측 포화
# ===============================
def fig_saturation():
    import torch
    import predictor as P
    P.load_ensemble()
    names = pd.read_csv(config.PROJECT_ROOT / "labels.csv").images[:40]

    def predict(bgr):
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (128, 128),
                         interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        rgbf = rgb.astype(np.float32) / 255.
        hsv[..., 0] /= 179.; hsv[..., 1] /= 255.; hsv[..., 2] /= 255.
        x = torch.from_numpy(np.transpose(np.concatenate([rgbf, hsv], 2), (2, 0, 1))).float().unsqueeze(0)
        with torch.no_grad():
            return P._aggregate([P._predict_with_tta(m, x) for m in P._models])

    scales, xs, ys, es = [None, 8.0, 4.0, 2.0, 1.5, 1.2, 1.0], [], [], []
    for sc in scales:
        preds, fr = [], []
        for n in names:
            bgr = cv2.resize(cv2.imread(str(config.PROJECT_ROOT / "train_images" / n)),
                             (640, 640), interpolation=cv2.INTER_AREA)
            if sc is None:
                sub = bgr
            else:
                m = _mask(bgr); yy, xx = np.where(m > 0)
                if len(yy) < 50: continue
                cy, cx = (yy.min() + yy.max()) // 2, (xx.min() + xx.max()) // 2
                hh, hw = int((yy.max() - yy.min()) // 2 * sc), int((xx.max() - xx.min()) // 2 * sc)
                sub = bgr[max(0, cy - hh):min(640, cy + hh), max(0, cx - hw):min(640, cx + hw)]
                if sub.size == 0: continue
            fr.append((_mask(sub) > 0).mean() * 100)
            preds.append(predict(sub))
        xs.append(np.mean(fr)); ys.append(np.mean(preds)); es.append(np.std(preds))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(xs, ys, yerr=es, marker="o", color=C_WARN, capsize=4, lw=2)
    ax.axhline(12.5, color="k", ls=":", lw=1.5, label="클리핑 상한 12.5")
    ax.axvspan(0, 3, color=C_OLD, alpha=0.18, label="학습 데이터 (딸기 ≈1%)")
    ax.axvspan(55, 72, color=C_NEW, alpha=0.18, label="운영 입력 (딸기 ≈63%)")
    ax.set_xlabel("딸기가 프레임에서 차지하는 면적 (%)")
    ax.set_ylabel("기존 모델의 예측 Brix")
    ax.set_title("[문제 2] 딸기를 크게 보여줄수록 예측이 올라가 12.5에서 포화\n"
                 "→ 운영 입력에서는 247장 전부 12.49±0.06, 전부 S등급", fontsize=11)
    ax.legend(fontsize=9)
    _save(fig, "2_saturation.png")


# ===============================
# 3) 학습 전 / 후
# ===============================
def fig_before_after():
    import torch
    import predictor as P
    P.load_ensemble()
    labels = pd.read_csv(config.PROJECT_ROOT / "data" / "crops.csv")

    def predict(bgr):
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (128, 128),
                         interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        rgbf = rgb.astype(np.float32) / 255.
        hsv[..., 0] /= 179.; hsv[..., 1] /= 255.; hsv[..., 2] /= 255.
        x = torch.from_numpy(np.transpose(np.concatenate([rgbf, hsv], 2), (2, 0, 1))).float().unsqueeze(0)
        with torch.no_grad():
            return P._aggregate([P._predict_with_tta(m, x) for m in P._models])

    old_pred, old_true = [], []
    for name, brix in zip(labels.images, labels.brix):
        old_pred.append(predict(cv2.imread(str(config.PROJECT_ROOT / "data" / "crops" / name))))
        old_true.append(float(brix))
    old_pred, old_true = np.array(old_pred), np.array(old_true)

    cv_ = {r["label"]: r for r in json.load(open(RUNS / "crossval_session.json"))}["증강 없음"]
    new_pred, new_true = np.array(cv_["pred"]), np.array(cv_["true"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for a, (p, t, title, note) in zip(ax, [
        (old_pred, old_true, "학습 전 — 기존 모델에 crop 입력",
         "예측 {:.2f}±{:.2f}\n전부 S등급".format(old_pred.mean(), old_pred.std())),
        (new_pred, new_true, "학습 후 — crop 재학습 (세션 5-fold)",
         "MAE {:.3f}  R² {:.3f}\n예측 {:.2f}±{:.2f}".format(
             np.abs(new_pred - new_true).mean(),
             1 - ((new_pred - new_true) ** 2).sum() / ((new_true - new_true.mean()) ** 2).sum(),
             new_pred.mean(), new_pred.std()))]):
        a.scatter(t, p, s=18, alpha=0.5, color=C_NEW if "후" in title else C_OLD, edgecolor="none")
        lim = [2, 15]
        a.plot(lim, lim, "k--", lw=1, label="완벽한 예측")
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("실측 Brix"); a.set_title(title, fontsize=11)
        a.text(0.04, 0.96, note, transform=a.transAxes, va="top", fontsize=9,
               bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax[0].set_ylabel("예측 Brix")
    ax[0].axhline(12.5, color=C_WARN, ls=":", lw=1.5)
    fig.suptitle("[결과 1] crop 재학습으로 예측 포화가 해소됨", fontsize=13, y=1.0)
    _save(fig, "3_before_after.png")


# ===============================
# 4) 증강 비교 — 무작위 vs 세션
# ===============================
def fig_augment():
    rand = {r["label"]: r for r in json.load(open(RUNS / "crossval_random.json"))}
    sess = {r["label"]: r for r in json.load(open(RUNS / "crossval_session.json"))}
    labs = [l for l in rand if l in sess]
    short = [l.split(" (")[0].replace("기하 + 밝기 · ", "").replace(" + ", "+") for l in labs]
    x = np.arange(len(labs)); w = 0.38

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, key, ylab, title in [
        (ax[0], "mae", "MAE (낮을수록 좋음)", "예측 오차"),
        (ax[1], "r2", "R² (높을수록 좋음)", "설명력"),
        (ax[2], "pred_std", "예측 표준편차", "예측이 얼마나 움직이는가")]:
        a.bar(x - w/2, [rand[l]["pooled"][key] for l in labs], w, label="무작위 5-fold", color=C_OLD)
        a.bar(x + w/2, [sess[l]["pooled"][key] for l in labs], w, label="세션 5-fold", color=C_NEW)
        a.set_xticks(x); a.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
        a.set_ylabel(ylab); a.set_title(title, fontsize=11); a.legend(fontsize=8)
    # 분할마다 기준선(항상 학습 평균 답하기)이 다릅니다. 세션 분할은 분포 이동 때문에 훨씬 높습니다.
    for store, color, name in [(rand, C_OLD, "무작위"), (sess, C_NEW, "세션")]:
        b = store[labs[0]]["pooled"]["baseline_mae"]
        ax[0].axhline(b, color=color, ls=":", lw=1.4)
        ax[0].text(len(labs) - 0.45, b + 0.01, "{} 기준선 {:.2f}".format(name, b),
                   fontsize=8, va="bottom", ha="right", color=color)
    ax[0].set_ylim(0, sess[labs[0]]["pooled"]["baseline_mae"] * 1.12)
    ax[1].axhline(0, color="k", lw=1)
    # 예측이 실측만큼 흔들려야 정상입니다. 라벨의 실제 표준편차를 기준선으로 그립니다.
    label_std = pd.read_csv(config.PROJECT_ROOT / "data" / "crops.csv").brix.std()
    ax[2].axhline(label_std, color="k", ls="--", lw=1.2)
    ax[2].text(-0.4, label_std + 0.03, "실측 라벨의 표준편차 {:.2f}".format(label_std),
               fontsize=8, va="bottom")
    ax[2].set_ylim(0, label_std * 1.18)
    fig.suptitle("[결과 2] 증강은 무작위 분할에서 차이 없고, 세션 분할에서는 모두 나빠짐\n"
                 "(단 40 epoch 고정 예산 — 예측 편차가 줄어든 것은 과소적합 신호)",
                 fontsize=12, y=1.04)
    _save(fig, "4_augmentation.png")


# ===============================
# 5) 배치 간 / 배치 내 신호 분해
# ===============================
def fig_between_within():
    r = {x["label"]: x for x in json.load(open(RUNS / "crossval_session.json"))}["증강 없음"]
    pred, true = np.array(r["pred"]), np.array(r["true"])
    df = pd.read_csv(config.PROJECT_ROOT / "data" / "crops.csv")
    df["session"] = df.src_width.astype(str) + "x" + df.src_height.astype(str)
    lbl = np.array(sum([[f["fold"]] * int((df.session == f["fold"]).sum())
                        for f in r["per_fold"]], []))
    uniq = list(dict.fromkeys(lbl))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, s in enumerate(uniq):
        m = lbl == s
        ax[0].scatter(true[m], pred[m], s=16, alpha=0.55, color=colors[i],
                      label="{} (n={})".format(s, m.sum()), edgecolor="none")
    ax[0].plot([2, 15], [2, 15], "k--", lw=1)
    ax[0].set_xlabel("실측 Brix"); ax[0].set_ylabel("예측 Brix")
    ax[0].set_title("전체 — r = {:+.3f}".format(np.corrcoef(pred, true)[0, 1]), fontsize=11)
    ax[0].legend(fontsize=7)

    mp = np.array([pred[lbl == s].mean() for s in uniq])
    mt = np.array([true[lbl == s].mean() for s in uniq])
    for i, s in enumerate(uniq):
        ax[1].scatter(mt[i], mp[i], s=170, color=colors[i], edgecolor="k", zorder=3)
        ax[1].annotate(s, (mt[i], mp[i]), fontsize=7, xytext=(6, -10),
                       textcoords="offset points")
    ax[1].plot([6, 11.5], [6, 11.5], "k--", lw=1)
    ax[1].set_xlabel("배치의 실측 평균 Brix"); ax[1].set_ylabel("배치의 예측 평균 Brix")
    ax[1].set_title("배치 '간' — r = {:+.3f}  (잘함)".format(np.corrcoef(mp, mt)[0, 1]),
                    fontsize=11, color="#137333")

    pc, tc = pred.astype(float).copy(), true.astype(float).copy()
    for s in uniq:
        m = lbl == s
        pc[m] -= pc[m].mean(); tc[m] -= tc[m].mean()
    for i, s in enumerate(uniq):
        m = lbl == s
        ax[2].scatter(tc[m], pc[m], s=16, alpha=0.55, color=colors[i], edgecolor="none")
    ax[2].axhline(0, color="k", lw=1); ax[2].axvline(0, color="k", lw=1)
    lim = max(abs(tc).max(), abs(pc).max()) * 1.05
    ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim)
    ax[2].set_xlabel("배치 평균 대비 실측 편차"); ax[2].set_ylabel("배치 평균 대비 예측 편차")
    ax[2].set_title("배치 '내' — r = {:+.3f}  (못함)".format(np.corrcoef(pc, tc)[0, 1]),
                    fontsize=11, color=C_WARN)
    fig.suptitle("[결과 3] 신호는 배치 '간'에 있고 배치 '내'에는 거의 없다\n"
                 "선별기는 같은 배치를 한 알씩 가르므로, 실효 상관은 0.58이 아니라 0.22",
                 fontsize=12, y=1.05)
    _save(fig, "5_between_within.png")


# ===============================
# 6) 등급 혼동행렬 — 현재 vs 보정 임계값
# ===============================
def fig_grades():
    from grades import CURRENT_THRESHOLDS, calibrate_thresholds, grade_metrics, GRADES
    r = {x["label"]: x for x in json.load(open(RUNS / "crossval_session.json"))}["증강 없음"]
    pred, true = np.array(r["pred"]), np.array(r["true"])
    fit = calibrate_thresholds(pred, true)
    a = grade_metrics(pred, true, CURRENT_THRESHOLDS)
    b = grade_metrics(pred, true, fit, true_thresholds=CURRENT_THRESHOLDS)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for axis, m, th, name in [(ax[0], a, CURRENT_THRESHOLDS, "현재 임계값"),
                              (ax[1], b, fit, "보정 임계값")]:
        cm = np.array(m["confusion"])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(4), GRADES); axis.set_yticks(range(4), GRADES)
        axis.set_xlabel("예측 등급"); axis.set_ylabel("실측 등급")
        axis.grid(False)
        for i in range(4):
            for j in range(4):
                axis.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                          color="white" if norm[i, j] > 0.5 else "black")
        axis.set_title("{}  [{:.2f}, {:.2f}, {:.2f}]\n정확도 {:.1%}   C 재현율 {:.0%}".format(
            name, *th, m["accuracy"], m["recall"]["C"]), fontsize=10)
    fig.suptitle("[결과 4] 임계값 보정으로 등급 C가 살아남 (재현율 3% → 33%)",
                 fontsize=12, y=1.03)
    _save(fig, "6_grades.png")


# ===============================
# 7) Hue 끊김
# ===============================
def fig_hue():
    df = pd.read_csv(config.PROJECT_ROOT / "data" / "crops.csv")
    allh = []
    for n in df.images[:80]:
        bgr = cv2.imread(str(config.PROJECT_ROOT / "data" / "crops" / n))
        s = 128.0 / max(bgr.shape[:2])
        hsv = cv2.cvtColor(cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2HSV)
        m = (hsv[..., 1] > 70) & (hsv[..., 2] > 50)
        allh.append(hsv[..., 0][m])
    h = np.concatenate(allh)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(h, bins=90, range=(0, 180), color=C_WARN, edgecolor="none")
    ax[0].axvspan(0, 15, color=C_NEW, alpha=0.15)
    ax[0].axvspan(165, 180, color=C_NEW, alpha=0.15)
    ax[0].set_xlabel("OpenCV Hue (0~179)"); ax[0].set_ylabel("픽셀 수")
    ax[0].set_title("딸기 과육의 Hue는 0과 179 양 끝에 걸침\n{:.0%}가 ≤15, {:.0%}가 ≥165 — 둘 다 같은 빨강".format(
        (h <= 15).mean(), (h >= 165).mean()), fontsize=10)

    ax[1].bar(["Hue ≤15\n(빨강)", "Hue ≥165\n(같은 빨강)"],
              [h[h <= 15].mean() / 179, h[h >= 165].mean() / 179], color=[C_NEW, C_WARN])
    ax[1].set_ylabel("모델 입력값 (H / 179)"); ax[1].set_ylim(0, 1.1)
    for i, v in enumerate([h[h <= 15].mean() / 179, h[h >= 165].mean() / 179]):
        ax[1].text(i, v, "{:.2f}".format(v), ha="center", va="bottom", fontsize=12)
    ax[1].set_title("같은 색인데 입력값은 정반대", fontsize=10)
    fig.suptitle("[문제 3] 빨강이 Hue 경계에 걸쳐 입력이 불연속", fontsize=12, y=1.03)
    _save(fig, "7_hue.png")


FIGURES = {1: fig_area_fraction, 2: fig_saturation, 3: fig_before_after,
           4: fig_augment, 5: fig_between_within, 6: fig_grades, 7: fig_hue}


def main():
    p = argparse.ArgumentParser(description="측정 결과 시각화")
    p.add_argument("--only", nargs="*", type=int, choices=sorted(FIGURES))
    args = p.parse_args()
    for n in (args.only or sorted(FIGURES)):
        print("[{}] {}".format(n, FIGURES[n].__name__))
        FIGURES[n]()


if __name__ == "__main__":
    main()
