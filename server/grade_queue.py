"""등급 큐와 선별 명령 워커.

검출 쪽은 과일을 찾는 즉시 등급을 밀어 넣지만, 선별기는 컨베이어를 타고 온 과일이
실제로 도착한 시점에 움직여야 합니다. 그래서 등급을 큐에 쌓아 두고 일정 간격으로
한 개씩만 꺼내 아두이노로 보냅니다.

    worker.start()      # START 시 1회
    worker.add("S")     # 검출될 때마다
    worker.stop()       # END 시 1회

간격은 ``config.QUEUE_POLL_INTERVAL``이며, 컨베이어 속도와 선별기까지의 거리에 맞춰
조정합니다.
"""

import queue
import threading
import time

import _bootstrap  # noqa: F401  (sys.path 설정)
import config
from result_store import increment_grade
from serial_link import send_motor_command


def send_and_count(grade):
    """워커가 등급 하나를 꺼낼 때마다 호출됩니다: 선별 명령 전송 + 집계."""
    grade = str(grade).upper()

    if grade not in config.GRADES:
        print("[QUEUE] 알 수 없는 등급 무시: {!r}".format(grade))
        return

    try:
        send_motor_command(grade)
        print("[QUEUE] 선별 명령 '{}' 전송".format(grade))
    except Exception as e:
        # 전송이 실패해도 아래 집계는 그대로 진행합니다. 즉, 집계는 "검사한 개수"이지
        # "선별에 성공한 개수"가 아닙니다. 장치 응답 기반 집계는 후속 개선 항목입니다.
        print("[QUEUE] 선별 명령 '{}' 전송 실패: {}".format(grade, e))

    try:
        increment_grade(grade)
    except Exception as e:
        print("[QUEUE] 집계 실패 '{}': {}".format(grade, e))


class GradeQueueWorker:
    """등급을 일정 간격으로 한 개씩 처리하는 단일 스레드 워커.

    동작 순서:

    1. 큐가 비어 있으면 대기합니다 (CPU를 쓰지 않습니다).
    2. 첫 항목이 들어온 시점부터 ``poll_interval``초 뒤에 한 개를 처리합니다.
    3. 큐에 남은 게 있으면 ``poll_interval``초마다 한 개씩 계속 처리합니다.
    4. 큐가 비면 다시 1번으로 돌아가 대기합니다.

    제약: 스레드는 한 번만 시작할 수 있어 ``stop()`` 후 같은 객체를 다시
    ``start()``할 수 없고, ``stop()`` 시 큐에 남은 항목은 처리되지 않고 버려집니다.
    """

    def __init__(self, poll_interval=None, handler=None):
        self.poll_interval = float(
            config.QUEUE_POLL_INTERVAL if poll_interval is None else poll_interval
        )
        self._handler = handler if handler is not None else send_and_count
        self._queue = queue.Queue()
        self._has_data = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._start_lock = threading.Lock()

    # ---- 외부에서 쓰는 메서드 ----

    def start(self):
        """워커 스레드를 시작합니다. 두 번 호출해도 안전합니다."""
        with self._start_lock:
            if self._started:
                return
            self._thread.start()
            self._started = True
            print("[QUEUE] 워커 시작 (간격 {}초)".format(self.poll_interval))

    def stop(self, timeout=3.0):
        """워커를 멈춥니다. 큐에 남은 항목은 처리되지 않습니다."""
        self._stopping.set()
        self._has_data.set()  # 대기 중이면 깨워서 루프를 빠져나오게 합니다.
        if self._started:
            self._thread.join(timeout=timeout)
        remaining = self._queue.qsize()
        if remaining:
            print("[QUEUE] 워커 종료 (처리하지 못한 {}건 폐기)".format(remaining))
        else:
            print("[QUEUE] 워커 종료")

    def add(self, grade):
        """등급을 큐에 넣습니다. 검출 쪽에서 호출하는 유일한 진입점입니다."""
        self._queue.put(grade)
        if self._queue.qsize() == 1:
            # 비어 있던 큐에 첫 항목이 들어왔으니 대기 중인 워커를 깨웁니다.
            self._has_data.set()

    # ---- 내부 ----

    def _run(self):
        while not self._stopping.is_set():
            # 1) 첫 항목이 들어올 때까지 대기
            self._has_data.wait()
            if self._stopping.is_set():
                break
            self._has_data.clear()

            # 2) 첫 항목 감지 시점을 기준으로 틱을 잡습니다.
            next_tick = time.monotonic() + self.poll_interval

            while not self._stopping.is_set():
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

                # 틱마다 정확히 한 개만 꺼냅니다.
                try:
                    grade = self._queue.get_nowait()
                except queue.Empty:
                    break  # 큐가 비었으니 다시 대기 상태로

                try:
                    self._handler(grade)
                finally:
                    self._queue.task_done()

                if self._queue.qsize() == 0:
                    break  # 더 없으면 대기 상태로
                next_tick += self.poll_interval


#: 제어 코드와 검출 이벤트 리더가 공유하는 전역 워커.
worker = GradeQueueWorker()


if __name__ == "__main__":
    # 큐 타이밍 확인용 데모. 시리얼이 없으면 전송 실패 로그만 남고 집계는 그대로 돕니다.
    from result_store import get_final_counts, reset_counts
    from serial_link import close_serial, init_serial

    try:
        init_serial()
    except Exception as e:
        print("[DEMO] 시리얼 없이 진행합니다:", e)

    reset_counts()
    worker.start()
    for delay, grade in [(1, "S"), (2, "A"), (4, "S"), (2, "S")]:
        time.sleep(delay)
        worker.add(grade)
        print("[DEMO] enqueue {}".format(grade))

    time.sleep(worker.poll_interval * 4)
    print("[DEMO] 최종 집계:", get_final_counts())

    worker.stop()
    close_serial()
