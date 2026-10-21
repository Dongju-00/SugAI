import os
import glob
import cv2
import time
import torch
import numpy as np
import pandas as pd

# ===============================
# 1. 데이터셋 클래스
# ===============================
class RGBHSVDataset_Lite(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_size=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _ensure_uint8_hwc(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
        else:
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        brix = float(self.data.iloc[idx, 1])
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        if self.img_size is not None:
            bgr = cv2.resize(bgr, self.img_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            out = self.transform(image=rgb)
            rgb = out["image"]
        rgb = self._ensure_uint8_hwc(rgb)
        rgb_f = rgb.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1] /= 255.0
        hsv[..., 2] /= 255.0
        combined = np.concatenate([rgb_f, hsv], axis=-1)
        combined = np.transpose(combined, (2, 0, 1))
        image_tensor = torch.from_numpy(combined).float()
        return image_tensor, torch.tensor(brix, dtype=torch.float32)

# ===============================
# 2. 모델 정의
# ===============================
class BrixRegression6CH_Deep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch, p_drop):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout(p_drop)
            )
        self.layer1 = conv_block(6, 32, 0.1)
        self.layer2 = conv_block(32, 64, 0.2)
        self.layer3 = conv_block(64, 128, 0.3)
        self.layer4 = conv_block(128, 256, 0.4)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x.squeeze(1)

# ===============================
# 3. 유틸 함수
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)
LO_CLIP, HI_CLIP = 3.0, 12.5
STD_TH1, STD_TH2 = 1.0, 2.0

image_folder = os.path.expanduser("~/brixproject/captures")
weight_paths = sorted(glob.glob(os.path.expanduser("~/brixproject/brix_model/best_model(Regressor)_fold*.pth")))
assert len(weight_paths) > 0, "❌ 모델 가중치를 찾을 수 없습니다."

# ✅ Jetson 안전 로드
models = []
for wp in weight_paths:
    print(f"🔍 로드 시도: {wp}")
    state = torch.load(wp, map_location="cpu")  # 1. CPU 로드
    m = BrixRegression6CH_Deep()
    missing, unexpected = m.load_state_dict(state, strict=False)  # 2. mismatch 체크
    if missing or unexpected:
        print(f"⚠️ Missing keys: {missing}")
        print(f"⚠️ Unexpected keys: {unexpected}")
    # 3. forward 더미 테스트
    try:
        m.eval()
        dummy = torch.randn(1, 6, 128, 128)
        _ = m(dummy)
        print("✅ forward 테스트 통과")
    except Exception as e:
        print(f"❌ forward 실패: {e}")
    models.append(m)

def preprocess_image(img_path):
    rgb = cv2.imread(img_path)
    if rgb is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb = rgb.astype(np.float32) / 255.0
    hsv[..., 0] /= 179.0
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    combined = np.concatenate([rgb, hsv], axis=2)
    combined = np.transpose(combined, (2, 0, 1))
    return torch.from_numpy(combined).float().unsqueeze(0)  # CPU 텐서

@torch.no_grad()
def predict_tta(model, x):
    # 안전한 Horizontal Flip (Jetson에서 호환)
    x_flip = torch.flip(x, dims=[3]).contiguous()
    y1 = model(x)
    y2 = model(x_flip)
    return (y1 + y2) / 2.0

def _trimmed_mean(arr: np.ndarray, trim_ratio: float = 0.2) -> float:
    k = int(arr.size * trim_ratio)
    if k == 0:
        return float(np.mean(arr))
    arr_sorted = np.sort(arr)
    return float(np.mean(arr_sorted[k:arr.size - k]))

def robust_aggregate(per_model, std_th1=STD_TH1, std_th2=STD_TH2,
                     trim_ratio=0.2, clip_lo=LO_CLIP, clip_hi=HI_CLIP):
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
    final_clipped = float(np.clip(final_raw, clip_lo, clip_hi))
    return {"final_clipped": final_clipped}

def brix_to_grade(b):
    if b < 6.05:
        return "C"
    elif b < 6.85:
        return "B"
    elif b < 7.75:
        return "A"
    else:
        return "S"

# ===============================
# 4. 추론 실행 (진행률 + 남은 시간)
# ===============================
with torch.no_grad():
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    total_imgs = len(image_paths)
    if total_imgs == 0:
        print("❌ 추론할 이미지가 없습니다.")
    else:
        start_time = time.time()
        for idx, path in enumerate(image_paths, start=1):
            x = preprocess_image(path)  # CPU 텐서
            per_model = []
            for m in models:
                m_gpu = m.to(DEVICE)
                pred = predict_tta(m_gpu, x.to(DEVICE)).item()
                per_model.append(pred)
                m_gpu.to("cpu")
                torch.cuda.empty_cache()
            agg = robust_aggregate(per_model)
            final_val = agg["final_clipped"]
            grade = brix_to_grade(final_val)
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total_imgs - idx)
            print(f"[{idx}/{total_imgs}] {os.path.basename(path)} → {grade} "
                  f"(경과 {elapsed:.1f}s, 남은 {remaining:.1f}s)")
        total_time = time.time() - start_time
        print(f"\n✅ 전체 완료: {total_imgs}장, 총 {total_time:.2f}초, 평균 {total_time/total_imgs:.2f}초/장")

