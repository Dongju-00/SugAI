"""검사 흐름 제어.

START/END 버튼 하나가 건드리는 것들을 모아 둔 모듈입니다.

START: 로컬 API 서버 → 시리얼 → 등급 큐 워커 → 검출 프로세스 → 컨베이어 시작
END:   검출 프로세스 정지 → 워커 정지 → 집계 전송 → 컨베이어 정지 → 정리

검출 프로세스(``yolov5/detect_fruit.py``)는 별도 프로세스로 띄우고, 표준출력에 찍는
``GRADE:<등급>:<경로>`` 줄을 리더 스레드가 읽어 등급 큐에 넣습니다. 프로세스를 분리한
이유는 YOLO와 PyTorch 초기화가 Tkinter 메인 루프를 멈추게 하기 때문입니다.
"""

import os
import subprocess
import sys
import threading

import requests

import _bootstrap  # noqa: F401  (sys.path 설정)
import config
from grade_queue import worker
from result_store import get_final_counts, reset_counts
from serial_link import close_serial, init_serial, send_motor_command

#: 삭제 대상 캡처 이미지 확장자.
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

_api_process = None       # uvicorn 서브프로세스
_detect_process = None    # detect_fruit.py 서브프로세스
_reader_thread = None     # _detect_process의 stdout을 읽는 스레드
_stop_reader = threading.Event()


# ===============================
# 1) 로컬 FastAPI 서버
# ===============================
def start_api_server():
    """``api.py``를 uvicorn 서브프로세스로 띄웁니다."""
    global _api_process
    if _api_process is not None and _api_process.poll() is None:
        print("[CTRL] API 서버가 이미 실행 중입니다.")
        return

    # PATH에 uvicorn 실행파일이 없는 환경이 있어 python -m uvicorn으로 띄웁니다.
    cmd = [
        sys.executable, "-m", "uvicorn", "api:app",
        "--host", config.LOCAL_API_HOST,
        "--port", str(config.LOCAL_API_PORT),
    ]
    print("[CTRL] API 서버 실행:", " ".join(cmd))
    try:
        _api_process = subprocess.Popen(cmd, cwd=_SERVER_DIR)
    except FileNotFoundError:
        print("[CTRL] uvicorn 실행 실패. '{} -m pip install uvicorn'으로 설치하세요."
              .format(sys.executable))
        raise


def stop_api_server():
    """uvicorn 서브프로세스를 종료합니다."""
    global _api_process
    if _api_process is None:
        print("[CTRL] API 서버가 실행 중이 아닙니다.")
        return
    print("[CTRL] API 서버 종료")
    _api_process.terminate()
    _api_process = None


# ===============================
# 2) 검출 프로세스
# ===============================
def _read_detect_output(process):
    """검출 프로세스의 출력을 한 줄씩 읽어 등급 큐에 넣습니다.

    약속된 줄 형식:
      ``CAPTURE:<경로>``             crop 저장됨
      ``GRADE:<등급>:<경로>``        당도 추론 결과
      ``MODEL_ERROR:<경로>:<사유>``  추론 실패
      ``DETECT_DONE``                검출 루프 종료
    """
    for line in process.stdout:
        if _stop_reader.is_set():
            break
        line = (line or "").strip()
        if not line:
            continue

        if line.startswith("CAPTURE:"):
            print("[CAPTURE]", line.split(":", 1)[1])

        elif line.startswith("GRADE:"):
            try:
                _, grade, path = line.split(":", 2)
            except ValueError:
                print("[CTRL] GRADE 줄 파싱 실패:", line)
                continue
            grade = grade.strip().upper()
            if grade in config.GRADES:
                worker.add(grade)
                print("[CTRL] 등급 {} 큐 적재 ({})".format(grade, path))
            else:
                print("[CTRL] 알 수 없는 등급:", grade)

        elif line.startswith("MODEL_ERROR:"):
            print("[MODEL]", line)

        elif line.startswith("DETECT_DONE"):
            print("[CTRL] 검출 루프 종료")

        # 그 밖의 YOLO 로그는 흘려보냅니다. 디버깅할 때만 아래 줄을 켜세요.
        # else:
        #     print("[YOLO]", line)

    print("[CTRL] 출력 리더 종료")


def start_detect():
    """검출 프로세스를 띄우고 출력 리더 스레드를 시작합니다."""
    global _detect_process, _reader_thread

    if _detect_process is not None and _detect_process.poll() is None:
        print("[CTRL] 검출 프로세스가 이미 실행 중입니다.")
        return

    config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    # 지난 검사에서 남은 정지 신호가 있으면 바로 종료되므로 먼저 치웁니다.
    if config.STOPFILE.exists():
        config.STOPFILE.unlink()

    cmd = [
        sys.executable, str(config.DETECT_SCRIPT),
        "--weights", str(config.DETECT_WEIGHTS),
        "--source", config.CAMERA_SOURCE,
        "--imgsz", str(config.DETECT_IMG_SIZE), str(config.DETECT_IMG_SIZE),
        "--conf-thres", str(config.DETECT_CONF_THRES),
        "--max-det", str(config.DETECT_MAX_DET),
        "--save-dir", str(config.CAPTURES_DIR),
        "--stopfile", str(config.STOPFILE),
    ]
    if config.DETECT_HALF:
        cmd.append("--half")
    if config.DETECT_VIEW_IMG:
        cmd.append("--view-img")

    print("[CTRL] 검출 실행:", " ".join(cmd))
    _detect_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,  # Python 3.6 호환 (text=True 대신)
        bufsize=1,
    )

    _stop_reader.clear()
    _reader_thread = threading.Thread(
        target=_read_detect_output, args=(_detect_process,), daemon=True
    )
    _reader_thread.start()


def stop_detect():
    """정지 신호 파일과 SIGTERM으로 검출 프로세스를 정리합니다."""
    global _detect_process, _reader_thread

    if _detect_process is None:
        print("[CTRL] 검출 프로세스가 실행 중이 아닙니다.")
        return

    print("[CTRL] 검출 프로세스 종료 중...")
    # 1) 루프가 스스로 빠져나올 기회를 줍니다 (카메라·창을 정상적으로 닫게).
    try:
        config.STOPFILE.write_text("1")
    except Exception as e:
        print("[CTRL] 정지 신호 파일 생성 실패:", e)

    # 2) 그래도 안 끝나면 SIGTERM, 마지막엔 SIGKILL.
    _detect_process.terminate()
    try:
        _detect_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("[CTRL] 응답이 없어 강제 종료합니다.")
        _detect_process.kill()

    _stop_reader.set()
    if _reader_thread is not None:
        _reader_thread.join(timeout=3)

    _detect_process = None
    _reader_thread = None


# ===============================
# 3) 뒷정리
# ===============================
def clear_captures():
    """캡처 폴더의 이미지를 모두 지웁니다.

    주의: 분석용으로 남기려면 END를 누르기 전에 따로 복사해 두어야 합니다.
    """
    if not config.CAPTURES_DIR.is_dir():
        print("[CTRL] 캡처 경로가 없습니다:", config.CAPTURES_DIR)
        return

    deleted = 0
    for path in config.CAPTURES_DIR.iterdir():
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            path.unlink()
            deleted += 1
        except Exception as e:
            print("[CTRL] 삭제 실패 {}: {}".format(path, e))
    print("[CTRL] 캡처 이미지 {}개 삭제".format(deleted))


def send_counts_to_api():
    """등급 집계를 로컬 API로 보냅니다 (API가 Spring으로 중계)."""
    payload = {"fruitTypeId": config.FRUIT_TYPE_ID, "counts": get_final_counts()}
    print("[CTRL] 집계 전송:", payload)
    try:
        res = requests.post(
            config.LOCAL_INSPECTION_URL, json=payload, timeout=config.HTTP_TIMEOUT
        )
        print("[CTRL] 응답 {}: {}".format(res.status_code, res.text[:200]))
    except Exception as e:
        print("[CTRL] 집계 전송 실패:", e)


# ===============================
# 4) START / END
# ===============================
def on_start_button_click():
    """START 버튼 처리."""
    print("[CTRL] === START ===")
    start_api_server()
    init_serial()
    worker.start()
    start_detect()

    send_motor_command(config.CONVEYOR_START_CMD, append_newline=False)
    print("[CTRL] 컨베이어 시작 신호 '{}' 전송".format(config.CONVEYOR_START_CMD))


def on_end_button_click():
    """END 버튼 처리."""
    print("[CTRL] === END ===")
    stop_detect()
    worker.stop()

    send_counts_to_api()

    try:
        send_motor_command(config.CONVEYOR_STOP_CMD, append_newline=False)
        print("[CTRL] 컨베이어 정지 신호 '{}' 전송".format(config.CONVEYOR_STOP_CMD))
    except Exception as e:
        print("[CTRL] 컨베이어 정지 신호 전송 실패:", e)

    reset_counts()
    close_serial()
    clear_captures()
    stop_api_server()
    print("[CTRL] 정리 완료")
