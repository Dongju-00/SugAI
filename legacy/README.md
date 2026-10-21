# legacy

실행 경로에서 **아무것도 import하거나 실행하지 않는** 파일들을 모아 둔 곳입니다.
삭제하지 않고 남긴 이유는 개발 과정의 기록이자, 나중에 비교·복구할 여지를 남기기 위함입니다.

**여기 있는 파일은 동작을 보장하지 않습니다.** 경로·모듈명이 리팩터링 이전 기준이라
그대로 실행하면 import 에러가 납니다. 참고용으로만 읽어 주세요.

현재 실제 실행 경로는 저장소 루트의 [README.md](../README.md)를 참고하세요.

## server/ — 구버전 제어 코드

| 파일 | 설명 | 현재 대체 |
|---|---|---|
| `main_control_ver2.py` | 초기 제어 로직. YOLO 프로세스 연동 전 단계로, 등급 큐와 FastAPI 전송만 수행 | `server/controller.py` |
| `main_control_ver3.py` | ver2에 YOLO 연동을 추가한 버전. `pysignal` 대신 매 전송마다 시리얼을 열고 닫음 | `server/controller.py` |
| `main_control_forTest.py` | 카메라·아두이노 없이 흐름만 확인하던 시연용 변형 | 없음 |
| `gui.py` | `main_control_ver3`을 호출하던 Tkinter 화면 | `server/app.py` |
| `gui_for_test.py` | `main_control_forTest`를 호출하던 테스트용 화면 | 없음 |

## yolov5/ — 구버전 검출 스크립트

| 파일 | 설명 | 현재 대체 |
|---|---|---|
| `detect_cap.py` | 첫 crop 저장 버전. SORT 추적 없이 프레임마다 저장 | `yolov5/detect_fruit.py` |
| `detect_cap2.py` | SORT 추적을 도입한 중간 버전. 등급 모델 호출은 없음 | `yolov5/detect_fruit.py` |
| `detect_cap4.py` | **CNN을 호출하지 않고** S/A/B/C를 순환 출력하던 시연용 경로. 실제 예측값이 아님 | 없음 |

## brix_model/ — 구버전 추론 스크립트

| 파일 | 설명 |
|---|---|
| `model.py` | 당도 추론 구버전. `brix_model/predictor.py`의 전신입니다. 학습 코드가 아닙니다 — 옵티마이저·손실 함수·학습 루프가 없습니다. |

**이 저장소에는 당도 모델의 학습 스크립트가 없습니다.** `brix_model/*.pth` 5개는 외부에서
학습해 가져온 결과물이고, 학습 설정(fold 분할, epoch, 증강, 손실)은 어디에도 기록되어
있지 않습니다. 재학습하려면 학습 코드를 새로 작성해야 합니다.

`model.py`에 남은 학습 흔적은 `RGBHSVDataset_Lite` 클래스 하나입니다. `labels.csv`
(`images,brix`)를 읽어 6채널 텐서를 만드는 Dataset으로, 정의만 있고 이 파일에서도
쓰이지 않습니다. 학습 코드를 다시 쓸 때 전처리 참고용으로 쓸 수 있습니다.

현재 버전과의 차이는 루트 [README.md](../README.md)의 '당도 예측 모델' 절과 아래 표를
참고하세요.

| 항목 | 구버전 `model.py` | 현재 `predictor.py` |
|---|---|---|
| 구조 | import만 해도 가중치 로드와 추론이 바로 실행됨 | `load_ensemble()` / `predict_brix()` 함수로 분리 |
| 모델 상주 | 이미지마다 모델을 GPU에 올리고 다시 CPU로 내림 | 한 번 올려 두고 재사용 |
| 등급 임계값 | C<6.05, B<6.85, A<7.75, S | C<5.56, B<6.97, A<9.23, S |
| 집계 반환 | `{"final_clipped": float}` dict | `float` |
| 출력 | 등급만 출력 | 당도와 등급 모두 출력 |
| 가중치 키 불일치 | 경고만 출력하고 계속 진행 | 예외를 던져 중단 |

## tools/

| 파일 | 설명 |
|---|---|
| `sub_motor_control.py` | 아두이노 수동 점검용 스크립트. 터미널에서 `S/A/B/C`(서보 각도), `1/0`(컨베이어)을 직접 입력해 전송합니다. 포트가 `/dev/ttyACM0`으로 하드코딩되어 있고, import 시점에 시리얼을 열기 때문에 아두이노가 없으면 바로 예외가 납니다. |
