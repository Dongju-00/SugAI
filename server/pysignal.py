# pysignal.py
import sys
import time
import serial
import threading

DEFAULT_BAUD = 9600
DEFAULT_PORT_LINUX = "/dev/ttyACM0"  # 필요 시 /dev/ttyUSB0
DEFAULT_PORT_WIN = "COM3"

_ser = None
_lock = threading.Lock()

def _default_port():
    return DEFAULT_PORT_WIN if sys.platform.startswith("win") else DEFAULT_PORT_LINUX

def init_serial(port: str = None, baud: int = DEFAULT_BAUD, wait_reset: float = 2.0):
    """앱 시작 시 1회 호출해서 포트를 열어 둔다."""
    global _ser
    if _ser and _ser.is_open:
        return _ser
    if port is None:
        port = _default_port()
    _ser = serial.Serial(port=port, baudrate=baud, timeout=1, write_timeout=1)
    if wait_reset > 0:
        time.sleep(wait_reset)  # Arduino 자동 리셋 대기
    return _ser

def close_serial():
    """앱 종료 시 1회 닫기."""
    global _ser
    if _ser:
        try: _ser.close()
        except Exception: pass
        _ser = None

def send_motor_command(cmd: str, append_newline: bool = True):
    """워커 handler에서 호출. 단일 문자 전송."""
    global _ser
    if _ser is None or not _ser.is_open:
        raise RuntimeError("Serial not initialized. Call init_serial() first.")
    if not isinstance(cmd, str) or len(cmd) != 1:
        raise ValueError(f"single char expected, got: {cmd!r}")
    data = (cmd + ("\n" if append_newline else "")).encode("utf-8")
    with _lock:
        _ser.write(data)
        _ser.flush()
