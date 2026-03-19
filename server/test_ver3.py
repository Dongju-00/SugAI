# test_ver2.py
import time
import threading
import queue
from typing import Any, Callable

# ① 아두이노 전송 함수 (시리얼 초기화는 main_control에서 1회 호출)
from pysignal import send_motor_command
# ② 카운트 증가 함수 (S/A/B/C 집계)
from result_store import increment_grade

VALID_GRADES = {"S", "A", "B", "C"}


def motor_and_count_handler(item: Any):
    """
    - 단건 폴링마다 호출
    - 1) pysignal로 해당 문자 전송
    - 2) 딕셔너리 카운트 증가
    """
    grade = str(item).upper()

    if grade not in VALID_GRADES:
        print(f"[WARN] Unknown grade '{item}' ignored.")
        return

    # 1) 모터/서보 명령 전송
    try:
        send_motor_command(grade)   # 단일 문자('S','A','B','C') 전송
        print(f"[HANDLER] Servo command '{grade}' sent.")
    except Exception as e:
        print(f"[ERROR] send_motor_command('{grade}') failed: {e}")
        # 전송 실패 시 카운트를 건너뛰고 싶다면 아래 return 주석 해제
        # return

    # 2) 카운트 증가
    try:
        increment_grade(grade)
    except Exception as e:
        print(f"[ERROR] increment_grade('{grade}') failed: {e}")


class PollingWorker:
    """
    단건 폴링 워커:
    - 큐가 비면 대기
    - 첫 데이터가 들어온 '그 순간' 기준 5초 후 첫 폴링
    - 이후 5초 간격으로 '정확히 1개'씩 처리
    - 처리 후 큐가 비면 다시 대기
    - 새로 들어오면 다시 '5초 후'부터 재개
    """
    def __init__(self, poll_interval: float = 5.0, handler: Callable[[Any], None] = None):
        self.q = queue.Queue()
        self.poll_interval = float(poll_interval)
        self._has_data_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._handler = handler if handler is not None else self._default_handler
        self._started = False
        self._lock = threading.Lock()

    # ---- 외부 API ----
    def start(self):
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def stop(self, timeout: float = 3.0):
        self._stop_event.set()
        self._has_data_event.set()  # 깨워서 종료 루프 진입
        if self._started:
            self._thread.join(timeout=timeout)

    def add(self, item: Any):
        """
        다른 코드에서 이 메서드만 호출하면 됨.
        - 비어있던 큐에 첫 아이템이 들어오면 이벤트 set으로 워커 깨움
        """
        self.q.put(item)
        if self.q.qsize() == 1:
            self._has_data_event.set()

    # ---- 내부 ----
    def _default_handler(self, item: Any):
        print(f"[{time.strftime('%H:%M:%S')}] POLL -> {item}")

    def _run(self):
        while not self._stop_event.is_set():
            # 1) 큐가 비면 대기
            self._has_data_event.wait()
            if self._stop_event.is_set():
                break

            # 이벤트 일회성 사용
            self._has_data_event.clear()

            # 2) 첫 데이터 감지 시점 기준 5초 뒤 첫 폴링
            next_tick = time.monotonic() + self.poll_interval

            # 5초 간격 단건 폴링 루프
            while not self._stop_event.is_set():
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

                # 폴링 시점: 정확히 1개만 꺼냄
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    # 큐가 비었으면 다시 대기 상태로
                    break

                try:
                    self._handler(item)  # ← pysignal 전송 + 카운트 증가
                finally:
                    self.q.task_done()

                # 처리 후 큐가 비면 대기로 복귀
                if self.q.qsize() == 0:
                    break

                # 남아있다면 다음 틱으로 5초 추가
                next_tick += self.poll_interval


# ---- 전역 워커: 다른 코드에서 import 해서 사용 ----
worker = PollingWorker(poll_interval=7.0, handler=motor_and_count_handler)


# 선택: 단독 실행 데모(실제 5초 간격이라 시간이 걸림)
if __name__ == "__main__":
    from pysignal import init_serial, close_serial
    from result_store import get_final_counts, reset_counts

    try:
        init_serial(port="/dev/ttyACM0", baud=9600)  # 환경에 맞게 포트 조정 (/dev/ttyUSB0 등)
    except Exception as e:
        print("[WARN] Serial init failed (demo will still run without sending):", e)

    reset_counts()
    worker.start()
    print("Demo: t=1 'S', t=3 'A', t=7 'S', t=9 'S' → 6,11,16,21초에 각각 폴링 예상")
    time.sleep(1); worker.add('S')
    time.sleep(2); worker.add('A')
    time.sleep(4); worker.add('S')
    time.sleep(2); worker.add('S')

    time.sleep(22)
    print("Final counts:", get_final_counts())

    worker.stop()
    try:
        close_serial()
    except Exception:
        pass
