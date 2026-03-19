# ~/jetson/brixproject/brix_model/model2.py
# Python 3.6 호환 버전
# 6채널(RGB+HSV) CNN 앙상블 + H-Flip TTA + 강건 집계 + S/A/B/C 매핑
# 공개 API:
#   load_ensemble(weight_glob: Optional[str] = None) -> None
#   predict_brix(image_path: str) -> float
#   predict_grade(image_path: str) -> str  # 'S' | 'A' | 'B' | 'C'

import os
import glob
from typing import Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn

# ===============================
# 0) 전역 설정
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)  # (W, H)
LO_CLIP, HI_CLIP = 3.0, 12.5    # 예측값 클리핑 범위
STD_TH1, STD_TH2 = 1.0, 2.0     # 표준편차 기반 집계 전략 임계치
_DEFAULT_GLOB = "~/brixproject/brix_model/best_model(Regressor)_fold*.pth"

_MODELS = []  # type: List[nn.Module]   # 로드된 앙상블 모델들 (DEVICE 상주)


# ===============================
# 1) 모델 정의
# ===============================
class BrixRegression6CH_Deep(nn.Module):
    def __init__(self):
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
                nn.Dropout(p_drop)
            )

        self.layer1 = conv_block(6,   32, 0.1)
        self.layer2 = conv_block(32,  64, 0.2)
        self.layer3 = conv_block(64, 128, 0.3)
        self.layer4 = conv_block(128, 256, 0.4)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x.squeeze(1)  # (N,)


# ===============================
# 2) 전처리 / TTA / 집계
# ===============================
def _preprocess_image(img_path):
    # type: (str) -> torch.Tensor
    """이미지 파일 경로 -> (1, 6, H, W) float32 텐서 (CPU 상)"""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)  # (W,H) 주의
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    rgb = rgb.astype(np.float32) / 255.0
    hsv[..., 0] /= 179.0
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0

    combined = np.concatenate([rgb, hsv], axis=2)      # (H, W, 6)
    combined = np.transpose(combined, (2, 0, 1))       # (6, H, W)
    x = torch.from_numpy(combined).float().unsqueeze(0)  # (1, 6, H, W)
    return x  # CPU 텐서

@torch.no_grad()
def _predict_tta(model, x_cpu):
    # type: (nn.Module, torch.Tensor) -> float
    """단일 모델에 대해 H-Flip TTA 2뷰 평균."""
    x = x_cpu.to(DEVICE, non_blocking=True)
    x_flip = torch.flip(x, dims=[3]).contiguous()

    y1 = model(x)
    y2 = model(x_flip)
    y = (y1 + y2) / 2.0
    return float(y.item())

def _trimmed_mean(arr, trim_ratio=0.2):
    # type: (np.ndarray, float) -> float
    k = int(arr.size * trim_ratio)
    if k == 0:
        return float(np.mean(arr))
    arr_sorted = np.sort(arr)
    return float(np.mean(arr_sorted[k:arr.size - k]))

def _robust_aggregate(per_model,
                      std_th1=STD_TH1,
                      std_th2=STD_TH2,
                      trim_ratio=0.2,
                      clip_lo=LO_CLIP,
                      clip_hi=HI_CLIP):
    # type: (List[float], float, float, float, float, float) -> float
    """모델별 예측 리스트 → 강건 단일 값."""
    arr = np.asarray(per_model, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    tmean_val = _trimmed_mean(arr, trim_ratio)
    std_val = float(np.std(arr))

    if std_val >= std_th2:
        final_raw = median_val
    elif std_val >= std_th1:
        final_raw = tmean_val
    else:
        final_raw = mean_val

    return float(np.clip(final_raw, clip_lo, clip_hi))


# ===============================
# 3) API: 로드 / 예측
# ===============================
def load_ensemble(weight_glob=None):
    # type: (Optional[str]) -> None
    """
    앙상블 가중치를 한 번만 로드하여 _MODELS(DEVICE 상주)에 보관.
    GPU 사용 가능 시 CUDA로 로드/상주시켜 속도 향상.
    """
    global _MODELS
    if _MODELS:
        return  # 이미 로드됨

    pattern = weight_glob or _DEFAULT_GLOB
    pattern = os.path.expanduser(pattern)  # '~' 확장
    paths = sorted(glob.glob(pattern))
    assert len(paths) > 0, "❌ 가중치 파일을 찾지 못했습니다: {}".format(pattern)

    for wp in paths:
        print("🔍 로드 시도: {}".format(wp))
        m = BrixRegression6CH_Deep().to(DEVICE)
        state = torch.load(wp, map_location=DEVICE)
        # PyTorch 버전에 따라 state가 dict일 수도, model.state_dict()일 수도 있음
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing:
            print("⚠️ Missing keys: {}".format(missing))
        if unexpected:
            print("⚠️ Unexpected keys: {}".format(unexpected))
        m.eval()
        # 더미 추론으로 장치 상주 확인
        _ = m(torch.randn(1, 6, 128, 128, device=DEVICE))
        print("✅ forward 테스트 통과")
        _MODELS.append(m)

@torch.no_grad()
def predict_brix(image_path):
    # type: (str) -> float
    """이미지 한 장에 대한 당도(brix) 예측값."""
    if not _MODELS:
        load_ensemble()
    x_cpu = _preprocess_image(image_path)
    per_model = [_predict_tta(m, x_cpu) for m in _MODELS]
    return _robust_aggregate(per_model)

def brix_to_grade(b):
    # type: (float) -> str
    if b < 5.56:
        return "C"
    elif b < 6.97:
        return "B"
    elif b < 9.23:
        return "A"
    else:
        return "S"

@torch.no_grad()
def predict_grade(image_path):
    # type: (str) -> str
    return brix_to_grade(predict_brix(image_path))

