"""제어 화면 (진입점).

    python app.py

START/END 버튼 두 개로 검사를 시작하고 끝냅니다. 실제 동작은 모두
``controller.py``에 있고 이 파일은 화면만 그립니다.

참고: 한 번 END를 누른 뒤 같은 창에서 다시 START를 누르면 등급 큐 워커가 재시작되지
않습니다 (스레드를 다시 시작할 수 없음). 다음 검사는 창을 다시 띄워서 시작하세요.
"""

import tkinter as tk

from controller import on_end_button_click, on_start_button_click


def run_gui():
    root = tk.Tk()
    root.title("SugAI 제어 패널")
    root.geometry("300x200")

    tk.Label(root, text="과일 분류 시스템", font=("Arial", 16)).pack(pady=10)

    tk.Button(
        root, text="START", width=20, height=2,
        bg="green", fg="white", command=on_start_button_click,
    ).pack(pady=10)

    tk.Button(
        root, text="END", width=20, height=2,
        bg="red", fg="white", command=on_end_button_click,
    ).pack(pady=10)

    print("[APP] 제어 화면 실행")
    root.mainloop()
    print("[APP] 제어 화면 종료")


if __name__ == "__main__":
    run_gui()
