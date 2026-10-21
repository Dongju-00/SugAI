"""아두이노 시리얼 연결.

포트를 프로그램 시작 시 한 번 열어 두고 계속 재사용합니다. 명령마다 포트를 여닫으면
아두이노가 그때마다 리셋되어 2초씩 멈추기 때문입니다.

    init_serial()             # START 시 1회
    send_motor_command("S")   # 필요할 때마다
    close_serial()            # END 시 1회
"""

import threading
import time

import serial

import _bootstrap  # noqa: F401  (sys.path 설정)
import config

#: 열려 있는 단일 연결. 여러 스레드(큐 워커, GUI 스레드)가 공유합니다.
_serial = None
#: 동시 write로 바이트가 섞이지 않도록 보호합니다.
_write_lock = threading.Lock()


def init_serial(port=None, baud=None, reset_wait=None):
    """시리얼 포트를 엽니다. 이미 열려 있으면 그 연결을 그대로 돌려줍니다."""
    global _serial
    if _serial is not None and _serial.is_open:
        return _serial

    port = port or config.SERIAL_PORT
    baud = baud or config.SERIAL_BAUD
    reset_wait = config.SERIAL_RESET_WAIT if reset_wait is None else reset_wait

    _serial = serial.Serial(port=port, baudrate=baud, timeout=1, write_timeout=1)
    if reset_wait > 0:
        # 포트를 열면 아두이노가 자동 리셋되므로, 부팅이 끝날 때까지 기다립니다.
        time.sleep(reset_wait)
    print("[SERIAL] {} @ {}bps 연결".format(port, baud))
    return _serial


def close_serial():
    """포트를 닫습니다. 열려 있지 않으면 아무 일도 하지 않습니다."""
    global _serial
    if _serial is None:
        return
    try:
        _serial.close()
        print("[SERIAL] 연결 종료")
    except Exception as e:
        print("[SERIAL] 종료 중 오류:", e)
    finally:
        _serial = None


def send_motor_command(cmd, append_newline=True):
    """단일 문자 명령을 아두이노로 보냅니다.

    펌웨어가 ``Serial.read()``로 한 글자씩 처리하므로 명령은 반드시 한 글자입니다.
    등급은 ``S/A/B/C``, 컨베이어는 ``1``(시작)/``0``(정지)입니다.
    """
    if _serial is None or not _serial.is_open:
        raise RuntimeError("시리얼이 열려 있지 않습니다. init_serial()을 먼저 호출하세요.")
    if not isinstance(cmd, str) or len(cmd) != 1:
        raise ValueError("한 글자 명령이어야 합니다: {!r}".format(cmd))

    data = (cmd + ("\n" if append_newline else "")).encode("utf-8")
    with _write_lock:
        _serial.write(data)
        _serial.flush()
