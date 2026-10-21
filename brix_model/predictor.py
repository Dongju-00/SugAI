"""당도(Brix) 추론.

과일 이미지 한 장을 받아 당도를 예측하고 S/A/B/C 등급으로 바꿉니다.

    load_ensemble()                 # 1회 (무거운 초기화)
    predict_brix("fruit.jpg")       # -> 8.41
    predict_grade("fruit.jpg")      # -> "A"

구성:
  * 입력은 RGB 3채널 + HSV 3채널을 합친 6채널 128x128 이미지입니다.
  * fold별로 학습한 모델 5개의 앙상블이고, 각 모델은 원본과 좌우 반전(TTA) 예측을
    평균합니다.
  * 모델 간 예측이 흔들릴수록 이상치에 둔한 집계로 바꿉니다 (평균 → 절사 평균 → 중앙값).

Jetson의 Python 3.6에서 돌아가야 하므로 그 이상의 문법은 쓰지 않습니다.
"""

import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (위에서 sys.path를 잡은 뒤에 import)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: 학습 때 쓴 입력 크기 (W, H). 바꾸면 가중치와 맞지 않습니다.
IMG_SIZE = (128, 128)

#: 예측값을 가둘 범위. 학습 데이터의 당도 분포를 벗어난 값을 쳐냅니다.
BRIX_MIN, BRIX_MAX = 3.0, 12.5

#: 모델 간 예측 표준편차 임계치. 이 값을 넘을수록 둔한 집계를 씁니다.
STD_TRIMMED_MEAN_TH = 1.0   # 이 이상이면 절사 평균
STD_MEDIAN_TH = 2.0         # 이 이상이면 중앙값
TRIM_RATIO = 0.2            # 절사 평균에서 양 끝에서 덜어낼 비율

#: 등급 경계 (예측 Brix 기준). C < 5.56 <= B < 6.97 <= A < 9.23 <= S
GRADE_B_MIN = 5.56
GRADE_A_MIN = 6.97
GRADE_S_MIN = 9.23

#: load_ensemble()이 채우는 모델 목록. DEVICE에 상주합니다.
_models = []  # type: List[nn.Module]


# ===============================
# 1) 모델 정의
# ===============================
class BrixRegression6CH_Deep(nn.Module):
    """6채널 입력을 받아 당도 하나를 회귀하는 CNN.

    ``in_channels``는 색 표현을 바꿔 실험할 때만 건드립니다. 기본값 6은 저장된
    가중치(``best_model(Regressor)_fold*.pth``)와 맞춰져 있으므로 바꾸면 그 가중치를
    더 이상 불러올 수 없습니다.
    """

    def __init__(self, in_channels=6):
        super(BrixRegression6CH_Deep, self).__init__()

        def conv_block(in_ch, out_ch, p_drop):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(p_drop),
            )

        # 깊어질수록 dropout을 키웁니다.
        self.layer1 = conv_block(in_channels, 32, 0.1)
        self.layer2 = conv_block(32, 64, 0.2)
        self.layer3 = conv_block(64, 128, 0.3)
        self.layer4 = conv_block(128, 256, 0.4)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 이름을 바꾸면 저장된 가중치의 키(fc.1.*, fc.4.*)와 어긋나 로드되지 않습니다.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x.squeeze(1)  # (N,)


# ===============================
# 2) 전처리 · TTA · 집계
# ===============================
def _preprocess_image(img_path):
    """이미지 경로 -> ``(1, 6, H, W)`` float32 CPU 텐서."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError("이미지를 읽을 수 없습니다: {}".format(img_path))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)  # cv2는 (W, H) 순서
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # 채널별 범위가 달라 각각 0~1로 맞춥니다 (OpenCV의 H는 0~179).
    rgb = rgb.astype(np.float32) / 255.0
    hsv[..., 0] /= 179.0
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0

    combined = np.concatenate([rgb, hsv], axis=2)   # (H, W, 6)
    combined = np.transpose(combined, (2, 0, 1))    # (6, H, W)
    return torch.from_numpy(combined).float().unsqueeze(0)


@torch.no_grad()
def _predict_with_tta(model, x_cpu):
    """한 모델의 원본·좌우반전 예측을 평균합니다."""
    x = x_cpu.to(DEVICE, non_blocking=True)
    x_flipped = torch.flip(x, dims=[3]).contiguous()
    y = (model(x) + model(x_flipped)) / 2.0
    return float(y.item())


def _trimmed_mean(values, trim_ratio=TRIM_RATIO):
    """양 끝에서 ``trim_ratio``씩 덜어낸 평균."""
    k = int(values.size * trim_ratio)
    if k == 0:
        return float(np.mean(values))
    ordered = np.sort(values)
    return float(np.mean(ordered[k:values.size - k]))


def _aggregate(per_model_predictions):
    """모델별 예측 리스트 -> 최종 당도 한 값.

    예측이 모여 있으면 평균, 흩어질수록 이상치에 둔한 집계로 넘어갑니다.
    """
    values = np.asarray(per_model_predictions, dtype=float)
    std = float(np.std(values))

    if std >= STD_MEDIAN_TH:
        aggregated = float(np.median(values))
    elif std >= STD_TRIMMED_MEAN_TH:
        aggregated = _trimmed_mean(values)
    else:
        aggregated = float(np.mean(values))

    return float(np.clip(aggregated, BRIX_MIN, BRIX_MAX))


# ===============================
# 3) 공개 API
# ===============================
def load_ensemble(weight_glob=None):
    """앙상블 가중치를 DEVICE에 올립니다. 이미 로드했으면 아무 일도 하지 않습니다."""
    if _models:
        return

    pattern = os.path.expanduser(weight_glob or config.BRIX_WEIGHT_GLOB)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("가중치 파일을 찾지 못했습니다: {}".format(pattern))

    for weight_path in paths:
        print("[BRIX] 로드:", weight_path)
        model = BrixRegression6CH_Deep().to(DEVICE)
        state = torch.load(weight_path, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # 여기서 넘어가면 해당 계층이 무작위 초기값으로 남아 엉뚱한 값을 예측합니다.
            raise RuntimeError(
                "가중치 구조가 모델과 다릅니다: {}\n  누락된 키: {}\n  알 수 없는 키: {}"
                .format(weight_path, list(missing), list(unexpected))
            )
        model.eval()
        # 더미 추론으로 장치 상주와 입력 형태를 미리 확인합니다.
        model(torch.randn(1, 6, IMG_SIZE[1], IMG_SIZE[0], device=DEVICE))
        _models.append(model)

    print("[BRIX] 앙상블 {}개 준비 완료 ({})".format(len(_models), DEVICE))


@torch.no_grad()
def predict_brix(image_path):
    """이미지 한 장의 예측 당도."""
    if not _models:
        load_ensemble()
    x_cpu = _preprocess_image(image_path)
    return _aggregate([_predict_with_tta(m, x_cpu) for m in _models])


def brix_to_grade(brix):
    """예측 당도를 등급 문자로 바꿉니다."""
    if brix < GRADE_B_MIN:
        return "C"
    if brix < GRADE_A_MIN:
        return "B"
    if brix < GRADE_S_MIN:
        return "A"
    return "S"


def predict_grade(image_path):
    """이미지 한 장의 예측 등급 (``"S"`` | ``"A"`` | ``"B"`` | ``"C"``)."""
    return brix_to_grade(predict_brix(image_path))
