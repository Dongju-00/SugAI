# ~/jetson/brixproject/brix_model/model2_test.py
# model2.py를 불러와서 단독 실행 가능한 테스트 스크립트
import os
import glob
from model2 import load_ensemble, predict_brix, predict_grade

def main():
    # 1) 모델 로드 (한 번만 실행)
    load_ensemble()

    # 2) 테스트할 이미지 폴더
    image_folder = os.path.expanduser("~/brixproject/captures")
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    if not image_paths:
        print("❌ 테스트할 이미지가 없습니다:", image_folder)
        return

    # 3) 이미지별 예측 수행
    for idx, path in enumerate(image_paths, start=1):
        try:
            brix = predict_brix(path)
            grade = predict_grade(path)
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)}")
            print(f"   → Brix: {brix:.2f}, Grade: {grade}")
        except Exception as e:
            print(f"⚠️ {path} 처리 실패: {e}")

if __name__ == "__main__":
    main()

