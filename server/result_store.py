"""등급별 검사 개수 집계.

한 번의 검사(START~END) 동안 S/A/B/C가 각각 몇 개 나왔는지만 세는 저장소입니다.
END 시점에 ``get_final_counts()``로 꺼내 Spring 서버로 보내고, ``reset_counts()``로
다음 검사를 준비합니다.
"""

import threading

import _bootstrap  # noqa: F401  (sys.path 설정)
import config

_lock = threading.Lock()
_grade_counts = {grade: 0 for grade in config.GRADES}


def increment_grade(grade):
    """해당 등급의 개수를 1 올립니다. 모르는 등급이면 무시합니다."""
    with _lock:
        if grade in _grade_counts:
            _grade_counts[grade] += 1


def get_final_counts():
    """현재 집계의 사본을 돌려줍니다 (호출 측에서 수정해도 원본은 안전)."""
    with _lock:
        return dict(_grade_counts)


def reset_counts():
    """모든 등급 개수를 0으로 되돌립니다."""
    with _lock:
        for grade in _grade_counts:
            _grade_counts[grade] = 0
