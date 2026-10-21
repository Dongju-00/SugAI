"""당도 회귀 학습용 데이터셋과 데이터 분할.

``predictor.py``와 **똑같은 전처리**(RGB/255, H/179, S/255, V/255, 6채널, 128x128)를
씁니다. 학습과 추론의 전처리가 갈라지면 모델이 운영에서 엉뚱하게 동작하므로,
전처리 코드는 여기 한 곳에만 둡니다.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from predictor import IMG_SIZE  # noqa: E402


#: 색 표현별 입력 채널 수. 모델의 in_channels와 맞춰야 합니다.
COLOR_SPACE_CHANNELS = {"hsv": 6, "huesincos": 7}


def to_tensor_image(rgb_uint8, color_space="hsv"):
    """RGB uint8 (H, W, 3) -> float32 (C, H, W).

    ``hsv``      : RGB + HSV 6채널. ``predictor.py``와 동일한 현재 방식.
    ``huesincos``: RGB + sin(색조) + cos(색조) + 채도 + 명도 7채널.

    ``hsv``의 문제: OpenCV의 Hue는 0~179로 원형이라 빨강이 양 끝에 걸칩니다.
    측정해 보면 딸기 과육 픽셀의 67%가 Hue<=15, 22%가 Hue>=165로 **같은 빨강인데**
    H/179 값이 0.02와 1.00으로 정반대가 됩니다. ``huesincos``는 색조를 각도로 보고
    sin/cos 두 채널로 풀어 이 끊김을 없앱니다.
    """
    rgb = cv2.resize(rgb_uint8, IMG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb_f = rgb.astype(np.float32) / 255.0

    if color_space == "hsv":
        hsv[..., 0] /= 179.0
        hsv[..., 1] /= 255.0
        hsv[..., 2] /= 255.0
        stacked = np.concatenate([rgb_f, hsv], axis=2)
    elif color_space == "huesincos":
        angle = np.deg2rad(hsv[..., 0] * 2.0)        # OpenCV Hue는 실제 각도의 절반
        stacked = np.concatenate([
            rgb_f,
            np.sin(angle)[..., None],
            np.cos(angle)[..., None],
            (hsv[..., 1] / 255.0)[..., None],
            (hsv[..., 2] / 255.0)[..., None],
        ], axis=2)
    else:
        raise ValueError("알 수 없는 color_space: {}".format(color_space))

    return np.transpose(stacked, (2, 0, 1))


def to_six_channel(rgb_uint8):
    """하위 호환용 별칭. ``predictor.py``와 동일한 6채널 전처리."""
    return to_tensor_image(rgb_uint8, "hsv")


# ===============================
# 증강
# ===============================
def augment_geometric(rgb, rng):
    """라벨을 바꾸지 않는 기하 변환. 당도는 딸기를 돌려도 변하지 않습니다."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    if rng.random() < 0.5:
        rgb = rgb[::-1, :]
    k = rng.integers(0, 4)
    if k:
        rgb = np.rot90(rgb, k)
    # 가장자리를 조금 잘라 박스 위치의 흔들림을 흉내 냅니다.
    h, w = rgb.shape[:2]
    cut = int(rng.integers(0, max(1, int(min(h, w) * 0.10))))
    if cut:
        rgb = rgb[cut:h - cut, cut:w - cut]
    return np.ascontiguousarray(rgb)


def augment_brightness(rgb, rng, strength=0.20):
    """밝기(명도)만 바꿉니다. 조명 차이에 강해지게 만드는 것이 목적입니다."""
    factor = 1.0 + rng.uniform(-strength, strength)
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment_color(rgb, rng, hue=6.0, sat=0.20):
    """색조와 채도를 바꿉니다.

    주의: 이 모델이 당도를 추정하는 단서가 바로 색입니다. 색을 흔들면
    "같은 색이 여러 당도에 대응한다"고 가르치게 되어 신호를 지울 수 있습니다.
    효과는 반드시 held-out 셋에서 확인하세요.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-sat, sat)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


AUGMENTATIONS = {
    "geometric": augment_geometric,
    "brightness": augment_brightness,
    "color": augment_color,
}


class BrixCropDataset(Dataset):
    """딸기 crop 이미지와 Brix 라벨.

    증강은 학습 셋에만 적용합니다. 검증·테스트에 증강을 걸면 점수가 의미를 잃습니다.
    """

    def __init__(self, images, brix, root, augments=(), seed=0, color_space="hsv"):
        self.images = list(images)
        self.brix = np.asarray(brix, dtype=np.float32)
        self.root = Path(root)
        self.augments = tuple(augments)
        self.color_space = color_space
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def __len__(self):
        return len(self.images)

    def _load(self, i):
        if i not in self._cache:
            bgr = cv2.imread(str(self.root / self.images[i]))
            if bgr is None:
                raise FileNotFoundError(self.root / self.images[i])
            # 증강 여지를 두되 메모리를 아끼려고 긴 변 192px로 맞춰 보관합니다.
            scale = 192.0 / max(bgr.shape[:2])
            if scale < 1.0:
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self._cache[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._cache[i]

    def __getitem__(self, i):
        rgb = self._load(i)
        for name in self.augments:
            rgb = AUGMENTATIONS[name](rgb, self.rng)
        x = to_tensor_image(rgb, self.color_space)
        return torch.from_numpy(x), torch.tensor(self.brix[i])


# ===============================
# 분할
# ===============================
def session_of(row):
    """촬영 세션 추정값. 원본 해상도가 같으면 같은 세션으로 봅니다."""
    return "{}x{}".format(row["src_width"], row["src_height"])


def stratified_split(df, ratios=(0.6, 0.2, 0.2), bins=8, seed=0):
    """Brix 구간을 맞춰 학습/검증/테스트로 나눕니다.

    단순 무작위 분할은 테스트 셋에 고당도나 저당도가 통째로 빠질 수 있습니다.
    다만 같은 세션의 이미지가 양쪽에 들어가므로 점수는 낙관적입니다.
    """
    rng = np.random.default_rng(seed)
    q = pd.qcut(df.brix, bins, labels=False, duplicates="drop")
    parts = [[], [], []]
    for b in np.unique(q):
        idx = np.array(df.index[q == b])
        rng.shuffle(idx)
        n1 = int(round(len(idx) * ratios[0]))
        n2 = int(round(len(idx) * (ratios[0] + ratios[1])))
        for k, chunk in enumerate((idx[:n1], idx[n1:n2], idx[n2:])):
            parts[k].extend(chunk)
    return [np.array(sorted(p)) for p in parts]


def session_split(df, test_sessions, val_ratio=0.2, seed=0):
    """지정한 세션을 통째로 테스트로 빼냅니다.

    촬영 세션(종이·조명·카메라)이 당도와 연관되어 있으면 모델이 딸기가 아니라
    세션 흔적을 외울 수 있습니다. 세션을 통째로 빼야 그 지름길을 막고
    '처음 보는 촬영 조건'에서의 성능을 잽니다.
    """
    rng = np.random.default_rng(seed)
    sess = df.apply(session_of, axis=1)
    test = df.index[sess.isin(test_sessions)].to_numpy()
    rest = np.array(df.index[~sess.isin(test_sessions)])
    rng.shuffle(rest)
    n_val = int(round(len(rest) * val_ratio))
    return np.array(sorted(rest[n_val:])), np.array(sorted(rest[:n_val])), np.array(sorted(test))
