# gui.py
import tkinter as tk
from main_control_ver3 import on_start_button_click, on_end_button_click

def run_gui():
    print("GUI 실행 시작")  # 로그 추가
    root = tk.Tk()
    root.title("Jetson 제어 패널")
    root.geometry("300x200")

    tk.Label(root, text=" 과일 분류 시스템", font=("Arial", 16)).pack(pady=10)

    start_btn = tk.Button(root, text="START", width=20, height=2, bg="green", fg="white", command=on_start_button_click)
    start_btn.pack(pady=10)


    end_btn = tk.Button(root, text="END", width=20, height=2, bg="red", fg="white", command=on_end_button_click)
    end_btn.pack(pady=10)

    print("GUI 실행 중...")  # 로그 추가
    root.mainloop()

if __name__ == "__main__":
    run_gui()

