import serial
import time

# 아두이노와 연결된 포트 확인 (실제 아두이노 포트로 수정 필요)
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Windows에서 연결된 포트 사용 (예: COM4)

def send_motor_command(command):
    # 아두이노로 모터 제어 명령 전송 (서보 모터 각도 제어 또는 스테퍼 모터 시작/정지)
    arduino.write(command.encode())  # 명령어를 문자열로 변환하여 전송

while True:
    # 사용자로부터 명령 입력 받기
    command = input("Enter 'S', 'A', 'B', 'C' for servo angles, or '1' to start, '0' to stop stepper motor, or 'q' to quit: ")

    if command == 'q':
        send_motor_command('0')
        send_motor_command('S')
        print("Exiting program.")
        break  # 프로그램 종료
    elif command in ['S', 'A', 'B', 'C']:
        send_motor_command(command)  # 서보 각도 제어 명령 전송
        print(f"Servo command '{command}' sent.")
    elif command in ['1', '0']:
        send_motor_command(command)  # 스테퍼 모터 시작/정지 명령 전송
        print(f"Stepper motor command '{command}' sent.")
    else:
        print("Invalid input. Please enter 'S', 'A', 'B', 'C' for servo, or '1'/'0' for stepper, or 'q' to quit.")

