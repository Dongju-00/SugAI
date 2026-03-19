# result_store.py

_grade_counts = {
    "S": 0,
    "A": 0,
    "B": 0,
    "C": 0
}

def increment_grade(grade: str):
    if grade in _grade_counts:
        _grade_counts[grade] += 1

def get_final_counts():
    return _grade_counts.copy()

def reset_counts():
    for key in _grade_counts:
        _grade_counts[key] = 0

