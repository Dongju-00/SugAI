# main_control.py
import subprocess
import requests
from result_store import get_final_counts, reset_counts
import sys, os, shutil, glob
import threading  # [ADD]

from pysignal import init_serial, close_serial, send_motor_command

# [ADD] 시리얼 전송용 import
import time
import serial
from serial.tools import list_ports

# 큐 관련 임포트
from test_ver3 import worker


FASTAPI_INSPECTION_URL = "http://localhost:8000/send-inspection"
FRUIT_TYPE_ID = 1  # 실제 상황에 맞게 변경 가능

# 서버 프로세스 변수
server_process = None

# [ADD] Arduino 포트 자동 탐색 + 전송 함수 (필요 최소)
ARDUINO_BAUD = 9600
ARDUINO_PORT_OVERRIDE = "/dev/ttyACM0"  # 특정 포트를 고정하고 싶으면 '/dev/ttyACM0' 등으로 설정

def _find_arduino_port():
    if ARDUINO_PORT_OVERRIDE:
        return ARDUINO_PORT_OVERRIDE
    prefer, fallback = [], []
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "cp210", "usb-serial", "silicon labs"]):
            prefer.append(p.device)
        else:
            fallback.append(p.device)
    return (prefer or fallback or [None])[0]

def _send_arduino(cmd: str):
    port = _find_arduino_port()
    if not port:
        print(" Arduino 포트를 찾지 못했습니다. USB 연결/권한을 확인하세요.")
        return
    try:
        with serial.Serial(port, ARDUINO_BAUD, timeout=1) as ser:
            time.sleep(2)  # Arduino가 시리얼 오픈 시 리셋되므로 잠깐 대기
            ser.write(cmd.encode())  # 아두이노는 Serial.read()로 단일 문자 처리
            ser.flush()
            print(f" Arduino({port}) ← '{cmd}' 전송")
    except Exception as e:
        print(" Arduino 전송 실패:", e)


def start_server():
    """FastAPI 서버 실행 (백그라운드에서)"""
    global server_process
    if server_process is None:
        print(" FastAPI 서버 실행 중...")

        # 디버그: 어떤 파이썬/uvicorn이 잡히는지 로그
        print(" sys.executable:", sys.executable)
        print(" which uvicorn:", shutil.which("uvicorn"))

        # main3.py가 있는 폴더에서 실행되게 보장 (경로 다르면 수정)
        server_dir = os.path.dirname(os.path.abspath(__file__))  # 예: /home/jetson/brixproject/server
        # server_dir = "/home/jetson/brixproject/server"  # 확실히 고정해도 됨

        # PATH 문제 회피: python -m uvicorn 으로 실행
        try:
            server_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "main3:app", "--host", "0.0.0.0", "--port", "8000"
            ], cwd=server_dir)
        except FileNotFoundError as e:
            print(" [ERR] uvicorn 실행 실패(FileNotFoundError).")
            print("  - 현재 파이썬:", sys.executable)
            print("  - which uvicorn:", shutil.which("uvicorn"))
            print("  - uvicorn이 설치되어 있는지와 Python 버전을 확인해 주세요.")
            raise
    else:
        print(" 이미 서버가 실행 중입니다.")


def stop_server():
    """FastAPI 서버 종료"""
    global server_process
    if server_process:
        print(" FastAPI 서버 종료 중...")
        server_process.terminate()
        server_process = None
    else:
        print(" 서버가 실행 중이 아닙니다.")


# === [ADD] YOLO(detect_cap3.py) 연동: 경로/프로세스/리더스레드/정리 ===
HOME = os.path.expanduser("~")
YOLO_PY      = os.path.join(HOME, "brixproject", "yolov5", "detect_cap4.py")
WEIGHTS      = os.path.join(HOME, "brixproject", "yolov5", "best.pt")
CAPTURES_DIR = os.path.join(HOME, "brixproject", "captures")
STOPFILE     = os.path.join(HOME, "brixproject", "stop_detect")
SOURCE       = (
    'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 '
    '! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
)

detect_process = None
reader_thread = None
stop_reader = threading.Event()

def _ensure_dirs():
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    # 시작 전에 stopfile 제거
    try:
        if os.path.exists(STOPFILE):
            os.remove(STOPFILE)
    except Exception:
        pass

def _stdout_reader(proc):
    """detect_cap3.py 표준출력 읽어서 CAPTURE/GRADE 이벤트 처리"""
    assert proc.stdout is not None
    for line in proc.stdout:
        if stop_reader.is_set():
            break
        s = (line or "").strip()
        if not s:
            continue
        if s.startswith("CAPTURE:"):
            path = s.split("CAPTURE:", 1)[1]
            print(f"[CAPTURE] {path}")
        elif s.startswith("GRADE:"):
            # GRADE:<grade>:<path>
            try:
                _, grade, path = s.split(":", 2)
                grade = grade.strip().upper()
            except ValueError:
                print("[PARSE] 잘못된 GRADE 라인:", s)
                continue
            if grade in ("S","A","B","C"):
                worker.add(grade)  # ← 큐에 적재(모터 전송+카운트 증가)
                print(f"[QUEUE] enqueue grade={grade}")
            else:
                print("[PARSE] 알 수 없는 등급:", grade)
        elif "MODEL_ERROR:" in s:
            print("[MODEL]", s)
        elif "DETECT_DONE" in s:
            print("[DETECT] done")
        else:
            # 필요 시 YOLO 로깅 확인
            # print("[YOLO]", s)
            pass
    print("[READER] 종료")

def start_yolo():
    """detect_cap3.py 실행 + stdout 파서 스레드 시작"""
    global detect_process, reader_thread, stop_reader
    if detect_process and detect_process.poll() is None:
        print(" detect_cap3.py가 이미 실행 중입니다.")
        return
    _ensure_dirs()

    cmd = [
        sys.executable, YOLO_PY,
        "--weights", WEIGHTS,
        "--source", SOURCE,
        "--imgsz", "640", "640",
        "--conf-thres", "0.6",
        "--max-det", "50",
        "--half",
        "--save-dir", CAPTURES_DIR,
        "--save-crop-once",
        "--stopfile", STOPFILE,
        "--view-img",  # GUI가 있을 때만 주석 해제
    ]
    print(" detect_cap3.py 실행:", " ".join(cmd))

    # brix_model을 못 찾는 경우 대비: PYTHONPATH에 추가
    env = os.environ.copy()
    model_dir = os.path.join(HOME, "brixproject", "brix_model")
    env["PYTHONPATH"] = (model_dir + os.pathsep + env.get("PYTHONPATH", ""))

    detect_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,  # Py3.6 호환 (text=True 대신)
        bufsize=1,
        env=env,
    )
    stop_reader.clear()
    reader_thread = threading.Thread(target=_stdout_reader, args=(detect_process,), daemon=True)
    reader_thread.start()

def stop_yolo():
    """stopfile + SIGTERM로 detect_cap3.py 그레이스풀 종료"""
    global detect_process, reader_thread
    if not detect_process:
        print(" detect_cap3.py가 실행 중이 아닙니다.")
        return
    print(" detect_cap3.py 종료 중...")
    try:
        # stopfile 신호
        with open(STOPFILE, "w") as f:
            f.write("1")
    except Exception:
        pass
    try:
        detect_process.terminate()
    except Exception:
        pass
    try:
        detect_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(" detect_cap3.py kill()")
        try:
            detect_process.kill()
        except Exception:
            pass
    stop_reader.set()
    if reader_thread:
        reader_thread.join(timeout=3)
    detect_process = None
    reader_thread = None

def _clean_captures():
    """CAPTURES_DIR 안의 이미지 파일 일괄 삭제"""
    try:
        patterns = ('*.jpg','*.jpeg','*.png','*.bmp','*.webp','*.JPG','*.JPEG','*.PNG','*.BMP','*.WEBP')
        deleted = 0
        if os.path.isdir(CAPTURES_DIR):
            for pat in patterns:
                for fp in glob.glob(os.path.join(CAPTURES_DIR, pat)):
                    try:
                        os.remove(fp); deleted += 1
                    except Exception as e:
                        print(f" 삭제 실패: {fp} ({e})")
            print(f" '{CAPTURES_DIR}' 내 이미지 {deleted}개 삭제 완료")
        else:
            print(f" 경로 없음: {CAPTURES_DIR}")
    except Exception as e:
        print(" 이미지 삭제 중 오류:", e)
# === [/ADD] ===

# 테스트용 함수
def on_conveyor_start_click():
    """컨베이어 시작 버튼"""
    try:
        send_motor_command('1', append_newline=False)
        print("컨베이어 시작 신호 '1' 전송")
    except Exception as e:
        print("컨베이어 시작 신호 전송 실패:", e)

def on_conveyor_stop_click():
    """컨베이어 종료 버튼"""
    try:
        send_motor_command('0', append_newline=False)
        print("컨베이어 정지 신호 '0' 전송")
    except Exception as e:
        print("컨베이어 정지 신호 전송 실패:", e)

def on_servo_S_click():
    """서보모터 S 버튼"""
    try:
        send_motor_command('S', append_newline=False)
        print("서보모터 'S' 전송")
    except Exception as e:
        print("서보모터 'S' 전송 실패:", e)

def on_servo_A_click():
    """서보모터 A 버튼"""
    try:
        send_motor_command('A', append_newline=False)
        print("서보모터 'A' 전송")
    except Exception as e:
        print("서보모터 'A' 전송 실패:", e)

def on_servo_B_click():
    """서보모터 B 버튼"""
    try:
        send_motor_command('B', append_newline=False)
        print("서보모터 'B' 전송")
    except Exception as e:
        print("서보모터 'B' 전송 실패:", e)

def on_servo_C_click():
    """서보모터 C 버튼"""
    try:
        send_motor_command('C', append_newline=False)
        print("서보모터 'C' 전송")
    except Exception as e:
        print("서보모터 'C' 전송 실패:", e)

def on_init_serial_click():
    """Init Serial 버튼"""
    try:
        init_serial()
        print("시리얼 초기화 완료")
    except Exception as e:
        print("시리얼 초기화 실패:", e)

def on_close_serial_click():
    """Close Serial 버튼"""
    try:
        close_serial()
        print("시리얼 종료 완료")
    except Exception as e:
        print("시리얼 종료 실패:", e)







def on_start_button_click():
    """START 버튼 클릭 시 실행될 함수"""
    start_server()

    # 시리얼 포트 열기
    init_serial(port="/dev/ttyACM0", baud=9600)
    # 큐 생성
    worker.start()

    # [ADD] YOLO 시작
    start_yolo()

    # [ADD] 컨베이어 시작 명령('1') 전송
    #_send_arduino('1')
    send_motor_command('1', append_newline=False)
    print("컨베이어 시작 신호 '1' 전송")



def on_end_button_click():
    """END 버튼 클릭 시 실행될 함수"""
    print(" END 버튼 클릭됨")

    # [ADD] YOLO 종료
    stop_yolo()

    # 큐 종료
    worker.stop()

    counts = get_final_counts()
    payload = {
        "fruitTypeId": FRUIT_TYPE_ID,
        "counts": counts
    }

    try:
        print(f" FastAPI로 전송: {payload}")
        res = requests.post(FASTAPI_INSPECTION_URL, json=payload, timeout=5)
        # .json()이 실패할 수 있어 짧게 텍스트만 출력
        print(" 응답:", res.status_code, str(res.text)[:200])
    except Exception as e:
        print(" 전송 실패:", e)

    # [ADD] 컨베이어 정지 명령('0') 전송
    #_send_arduino('0')
    send_motor_command('0', append_newline=False)
    print("컨베이어 정지 신호 '0' 전송")

    # 이후 상태 초기화
    reset_counts()

    # 시그널 포트 닫기
    close_serial()

    # 이미지 저장해둔 디렉토리 내 사진 다 지우기
    _clean_captures()
    # try:
    #     captures_dir = os.path.expanduser('~/brixproject/captures')
    #     patterns = ('*.jpg','*.jpeg','*.png','*.bmp','*.webp','*.JPG','*.JPEG','*.PNG','*.BMP','*.WEBP')
    #     deleted = 0
    #     if os.path.isdir(captures_dir):
    #         for pat in patterns:
    #             for fp in glob.glob(os.path.join(captures_dir, pat)):
    #                 try:
    #                     os.remove(fp)
    #                     deleted += 1
    #                 except Exception as e:
    #                     print(f" 삭제 실패: {fp} ({e})")
    #         print(f" '{captures_dir}' 내 이미지 {deleted}개 삭제 완료")
    #     else:
    #         print(f" 경로 없음: {captures_dir}")
    # except Exception as e:
    #     print(" 이미지 삭제 중 오류:", e)

    print(" 결과 초기화 완료")
    stop_server()  # 서버 종료
