
import tkinter as tk
from main_control_forTest import (
    on_start_button_click,
    on_end_button_click,
    on_conveyor_start_click,
    on_conveyor_stop_click,
    on_servo_S_click,
    on_servo_A_click,
    on_servo_B_click,
    on_servo_C_click,
    on_init_serial_click,    # [NEW]
    on_close_serial_click,   # [NEW]
)

def run_gui():
    print("GUI 실행 시작")
    root = tk.Tk()
    root.title("Jetson 제어 패널")
    root.geometry("360x380")

    tk.Label(root, text=" 과일 분류 시스템", font=("Arial", 16)).pack(pady=10)

    # START / END
    start_btn = tk.Button(root, text="START", width=20, height=2, bg="green", fg="white",
                          command=on_start_button_click)
    start_btn.pack(pady=6)

    end_btn = tk.Button(root, text="END", width=20, height=2, bg="red", fg="white",
                        command=on_end_button_click)
    end_btn.pack(pady=6)

    # 컨베이어 제어
    conveyor_frame = tk.Frame(root)
    conveyor_frame.pack(pady=10)
    tk.Label(conveyor_frame, text="컨베이어 제어", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=(0,8))

    conveyor_start_btn = tk.Button(conveyor_frame, text="컨베이어 시작", width=20, height=2,
                                   command=on_conveyor_start_click)
    conveyor_start_btn.grid(row=1, column=0, padx=5, pady=5)

    conveyor_stop_btn = tk.Button(conveyor_frame, text="컨베이어 종료", width=20, height=2,
                                  command=on_conveyor_stop_click)
    conveyor_stop_btn.grid(row=1, column=1, padx=5, pady=5)

    # 서보(등급) 제어
    servo_frame = tk.Frame(root)
    servo_frame.pack(pady=10)
    tk.Label(servo_frame, text="서보모터 등급", font=("Arial", 12)).grid(row=0, column=0, columnspan=4, pady=(0,8))

    btn_S = tk.Button(servo_frame, text="S", width=8, height=2, command=on_servo_S_click)
    btn_A = tk.Button(servo_frame, text="A", width=8, height=2, command=on_servo_A_click)
    btn_B = tk.Button(servo_frame, text="B", width=8, height=2, command=on_servo_B_click)
    btn_C = tk.Button(servo_frame, text="C", width=8, height=2, command=on_servo_C_click)

    btn_S.grid(row=1, column=0, padx=5, pady=5)
    btn_A.grid(row=1, column=1, padx=5, pady=5)
    btn_B.grid(row=1, column=2, padx=5, pady=5)
    btn_C.grid(row=1, column=3, padx=5, pady=5)


      # NEW: Serial 제어
    serial_frame = tk.Frame(root)
    serial_frame.pack(pady=10)
    tk.Label(serial_frame, text="시리얼 제어", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=(0,8))

    init_btn = tk.Button(serial_frame, text="Init Serial", width=16, height=2,
                         command=on_init_serial_click)
    init_btn.grid(row=1, column=0, padx=5, pady=5)

    close_btn = tk.Button(serial_frame, text="Close Serial", width=16, height=2,
                          command=on_close_serial_click)
    close_btn.grid(row=1, column=1, padx=5, pady=5)

    print("GUI 실행 중...")
    root.mainloop()

if __name__ == "__main__":
    run_gui()

