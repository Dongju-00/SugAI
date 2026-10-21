"""SugAI 실행 설정.

경로·장치·외부 서버 설정을 한곳에 모았습니다. 거의 모든 값은 환경변수로 덮어쓸 수
있으므로, 다른 장비로 옮길 때 소스를 고치지 않고 환경변수만 바꾸면 됩니다.

    export SUGAI_SERIAL_PORT=/dev/ttyUSB0
    export SUGAI_SPRING_BASE_URL=http://10.0.0.5:8080
    export SUGAI_SPRING_PASSWORD=...

프로젝트 경로는 이 파일의 위치에서 계산합니다. 따라서 저장소를 어디에 clone해도
그대로 동작하며, 예전처럼 ``~/brixproject``에 두어야 할 필요가 없습니다.

server/, brix_model/, yolov5/ 안의 스크립트는 다음과 같이 이 모듈을 불러옵니다.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
"""

import os
import sys
from pathlib import Path


def _env(name, default):
    """``SUGAI_`` 환경변수를 읽습니다. 비어 있으면 기본값을 씁니다."""
    return os.environ.get("SUGAI_" + name) or default


def _env_float(name, default):
    raw = os.environ.get("SUGAI_" + name)
    return float(raw) if raw else default


def _env_int(name, default):
    raw = os.environ.get("SUGAI_" + name)
    return int(raw) if raw else default


# ===============================
# 1) 경로
# ===============================
#: 저장소 루트. SUGAI_HOME으로 덮어쓸 수 있습니다.
PROJECT_ROOT = Path(_env("HOME_DIR", Path(__file__).resolve().parent)).expanduser()

YOLO_DIR = PROJECT_ROOT / "yolov5"
BRIX_MODEL_DIR = PROJECT_ROOT / "brix_model"

#: 검출 스크립트가 crop한 과일 이미지를 쌓아 두는 곳. END 시 비워집니다.
CAPTURES_DIR = Path(_env("CAPTURES_DIR", PROJECT_ROOT / "captures")).expanduser()

#: 이 파일이 만들어지면 검출 루프가 스스로 빠져나옵니다 (graceful stop 신호).
STOPFILE = Path(_env("STOPFILE", PROJECT_ROOT / "stop_detect")).expanduser()

#: 과일 검출 스크립트와 그 가중치. best.pt는 저장소에 포함되지 않으므로 따로 받아야 합니다.
DETECT_SCRIPT = YOLO_DIR / "detect_fruit.py"
DETECT_WEIGHTS = Path(_env("DETECT_WEIGHTS", YOLO_DIR / "best.pt")).expanduser()

#: 당도 회귀 앙상블 가중치 (fold 5개).
BRIX_WEIGHT_GLOB = _env(
    "BRIX_WEIGHT_GLOB", str(BRIX_MODEL_DIR / "best_model(Regressor)_fold*.pth")
)


# ===============================
# 2) 카메라 · 검출
# ===============================
#: Jetson CSI 카메라용 GStreamer 파이프라인. USB 웹캠이면 "0", 영상 파일이면 그 경로.
JETSON_CAMERA_PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 "
    "! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)
CAMERA_SOURCE = _env("CAMERA_SOURCE", JETSON_CAMERA_PIPELINE)

DETECT_IMG_SIZE = _env_int("DETECT_IMG_SIZE", 640)
DETECT_CONF_THRES = _env_float("DETECT_CONF_THRES", 0.6)
DETECT_MAX_DET = _env_int("DETECT_MAX_DET", 50)
#: True면 검출 화면을 띄웁니다. 디스플레이가 없는 환경에서는 꺼야 합니다.
DETECT_VIEW_IMG = _env("DETECT_VIEW_IMG", "1") not in ("0", "false", "False")
#: FP16 추론. Jetson GPU에서만 의미가 있습니다.
DETECT_HALF = _env("DETECT_HALF", "1") not in ("0", "false", "False")


# ===============================
# 3) 아두이노 시리얼
# ===============================
SERIAL_PORT = _env("SERIAL_PORT", "COM3" if sys.platform.startswith("win") else "/dev/ttyACM0")
SERIAL_BAUD = _env_int("SERIAL_BAUD", 9600)
#: 포트를 열면 아두이노가 리셋되므로 그만큼 기다립니다.
SERIAL_RESET_WAIT = _env_float("SERIAL_RESET_WAIT", 2.0)

#: 펌웨어와 맞춰야 하는 명령 규약.
GRADES = ("S", "A", "B", "C")
CONVEYOR_START_CMD = "1"
CONVEYOR_STOP_CMD = "0"

#: 등급 큐에서 한 개를 꺼내 선별 명령을 보내는 간격(초).
#: 컨베이어 속도와 선별기 위치에 맞춰 조정합니다.
QUEUE_POLL_INTERVAL = _env_float("QUEUE_POLL_INTERVAL", 7.0)


# ===============================
# 4) 로컬 FastAPI
# ===============================
LOCAL_API_HOST = _env("LOCAL_API_HOST", "0.0.0.0")
LOCAL_API_PORT = _env_int("LOCAL_API_PORT", 8000)
#: 제어 코드가 집계 결과를 POST하는 로컬 주소.
LOCAL_INSPECTION_URL = "http://localhost:{}/send-inspection".format(LOCAL_API_PORT)

#: 검사 대상 과일 종류 ID (Spring 서버의 fruitType과 맞춰야 합니다).
FRUIT_TYPE_ID = _env_int("FRUIT_TYPE_ID", 1)


# ===============================
# 5) 외부 Spring 서버
# ===============================
SPRING_BASE_URL = _env("SPRING_BASE_URL", "http://13.125.20.176:8080")
SPRING_LOGIN_URL = "{}/auth/login".format(SPRING_BASE_URL)
SPRING_INSPECTION_URL = "{}/inspections".format(SPRING_BASE_URL)

#: 연동 계정. 운영 환경에서는 반드시 환경변수로 넘기세요.
SPRING_USERNAME = _env("SPRING_USERNAME", "test1")
SPRING_PASSWORD = _env("SPRING_PASSWORD", "1234")

HTTP_TIMEOUT = _env_float("HTTP_TIMEOUT", 5.0)
