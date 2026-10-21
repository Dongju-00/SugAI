"""캡처 폴더의 이미지를 순회하며 당도·등급을 출력합니다.

    python predict_images.py [이미지_폴더]

폴더를 생략하면 ``config.CAPTURES_DIR``을 봅니다. 이미지별 추론을 눈으로 확인하는
도구이며, MAE/RMSE 같은 성능 지표를 계산하지는 않습니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (위에서 sys.path를 잡은 뒤에 import)
from predictor import brix_to_grade, load_ensemble, predict_brix  # noqa: E402


def main():
    image_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else config.CAPTURES_DIR
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        print("추론할 이미지가 없습니다:", image_dir)
        return 1

    load_ensemble()

    for i, path in enumerate(image_paths, start=1):
        try:
            # 추론은 한 번만 돌리고 등급은 그 값에서 바로 계산합니다.
            brix = predict_brix(str(path))
            print("[{}/{}] {} -> Brix {:.2f}, 등급 {}"
                  .format(i, len(image_paths), path.name, brix, brix_to_grade(brix)))
        except Exception as e:
            print("[{}/{}] {} -> 실패: {}".format(i, len(image_paths), path.name, e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
