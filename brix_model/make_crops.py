"""학습 이미지에서 딸기 박스 crop을 뽑아 학습용 데이터셋을 만듭니다.

    python make_crops.py [--src ../train_images] [--out ../data/crops]

왜 필요한가
-----------
원본 학습 이미지는 흰 종이 위에 딸기 한 알을 놓고 찍은 전체 사진이라, 딸기가
프레임의 약 1%만 차지합니다. 반면 운영 중에 모델이 받는 입력은 ``detect_fruit.py``가
YOLO 박스로 잘라낸 crop이라 딸기가 프레임의 60% 이상을 차지합니다.

이 차이 때문에 전체 사진으로 학습한 모델은 운영 입력에서 예측이 포화합니다.
이 스크립트는 학습 데이터를 운영 입력과 같은 모양으로 맞춰 줍니다.

딸기 검출에는 YOLO 대신 HSV 빨강 마스크를 씁니다. 배경이 흰 종이로 단순해서
충분히 안정적이고, 검출 가중치(best.pt) 없이도 돌릴 수 있기 때문입니다.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

#: 빨강은 Hue 0 근처에서 한 바퀴 돌기 때문에 두 구간을 따로 잡아 합칩니다.
_RED_LOW = ((0, 70, 50), (12, 255, 255))
_RED_HIGH = ((168, 70, 50), (179, 255, 255))

#: 박스를 이 비율만큼 넓혀 꼭지와 윤곽이 잘리지 않게 합니다.
BOX_MARGIN = 0.12
#: 마스크가 이 비율보다 작으면 검출 실패로 봅니다.
MIN_AREA_RATIO = 0.0002


def find_strawberry(bgr):
    """가장 큰 빨간 덩어리의 바운딩 박스 ``(x0, y0, x1, y1)``. 못 찾으면 ``None``."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *_RED_LOW) | cv2.inRange(hsv, *_RED_HIGH)

    # 잡티를 지우고(OPEN) 과육 내부의 구멍을 메웁니다(CLOSE).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < MIN_AREA_RATIO * bgr.shape[0] * bgr.shape[1]:
        return None

    x, y, w, h = (stats[biggest, c] for c in
                  (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    mx, my = int(w * BOX_MARGIN), int(h * BOX_MARGIN)
    return (max(0, x - mx), max(0, y - my),
            min(bgr.shape[1], x + w + mx), min(bgr.shape[0], y + h + my))


def crop_strawberry(bgr):
    """딸기 박스로 자른 이미지. 못 찾으면 ``None``."""
    box = find_strawberry(bgr)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1]
    return crop if crop.size else None


def main():
    parser = argparse.ArgumentParser(description="딸기 박스 crop 데이터셋 생성")
    parser.add_argument("--src", default=str(config.PROJECT_ROOT / "train_images"))
    parser.add_argument("--labels", default=str(config.PROJECT_ROOT / "labels.csv"))
    parser.add_argument("--out", default=str(config.PROJECT_ROOT / "data" / "crops"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.src)

    with open(args.labels) as f:
        rows = list(csv.DictReader(f))

    kept, failed, fractions = [], [], []
    for row in rows:
        name = row["images"]
        bgr = cv2.imread(str(src_dir / name))
        if bgr is None:
            failed.append((name, "읽기 실패"))
            continue

        # 원본이 4000px급이라 그대로 마스킹하면 느립니다. 축소본에서 박스를 찾고
        # 좌표를 원본 배율로 되돌려 원본 해상도에서 잘라냅니다.
        scale = 640.0 / max(bgr.shape[:2])
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        box = find_strawberry(small)
        if box is None:
            failed.append((name, "딸기를 찾지 못함"))
            continue

        x0, y0, x1, y1 = (int(v / scale) for v in box)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            failed.append((name, "빈 crop"))
            continue

        cv2.imwrite(str(out_dir / name), crop)
        kept.append({"images": name, "brix": row["brix"],
                     "width": crop.shape[1], "height": crop.shape[0],
                     "src_width": bgr.shape[1], "src_height": bgr.shape[0]})
        fractions.append(crop.shape[0] * crop.shape[1] / (bgr.shape[0] * bgr.shape[1]))

    with open(out_dir.parent / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
        w.writeheader()
        w.writerows(kept)

    sides = [min(k["width"], k["height"]) for k in kept]
    print("crop 생성 {}장 / 실패 {}장".format(len(kept), len(failed)))
    print("  원본 대비 crop 면적 비율: 평균 {:.2%}".format(float(np.mean(fractions))))
    print("  crop 짧은 변: 최소 {} / 중앙값 {} / 최대 {}px".format(
        min(sides), int(np.median(sides)), max(sides)))
    print("  저장 위치: {}".format(out_dir))
    for name, why in failed:
        print("  [실패] {}: {}".format(name, why))


if __name__ == "__main__":
    main()
