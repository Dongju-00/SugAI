from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import requests

app = FastAPI()

# ✅ Spring 서버 주소
SPRING_LOGIN_URL = "http://13.125.20.176:8080/auth/login"
SPRING_INSPECTION_URL = "http://13.125.20.176:8080/inspections"

# ✅ 로그인에 사용할 Jetson 계정 정보
USERNAME = "test1"        # 또는 email
PASSWORD = "1234"         # 실제 비밀번호


class InspectionData(BaseModel):
    fruitTypeId: int
    counts: Dict[str, int]  # 예: {"S": 30, "A": 10, "B": 3}


# ✅ accessToken 발급 요청 함수 (CommonResponse 구조 대응)
def get_access_token(username: str, password: str):
    payload = {
        "username": username,     # ❗ Spring이 "email"을 요구하면 여기 수정 필요
        "password": password
    }
    headers = {"Content-Type": "application/json"}

    try:
        res = requests.post(SPRING_LOGIN_URL, json=payload, headers=headers)
        res.raise_for_status()  # 응답 코드가 200이 아니면 예외 발생

        json_res = res.json()
        print("🧾 Spring 로그인 응답:", json_res)

        # ✅ CommonResponse 구조에서 data.accessToken 추출
        token = json_res.get("data", {}).get("accessToken")
        print("🔑 받은 accessToken:", token)
        return token
    except Exception as e:
        print("❌ 로그인 실패:", e)
        return None


# ✅ 실제 등급 검사 전송 엔드포인트
@app.post("/send-inspection")
def send_inspection(data: InspectionData):
    print("📥 [FastAPI] 요청 수신 완료")
    print("📦 받은 데이터:", data.dict())

    # 🔐 로그인 → 토큰 발급
    token = get_access_token(USERNAME, PASSWORD)

    if not token:
        return {"message": "AccessToken 발급 실패"}

    # 📦 검사 결과 전송
    payload = {
        "fruitTypeId": data.fruitTypeId,
        "counts": data.counts
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        print("🚀 Spring 서버로 POST 요청 전송 중...")
        response = requests.post(SPRING_INSPECTION_URL, json=payload, headers=headers)

        print("📨 Spring 응답 수신 완료")
        print("🔢 상태코드:", response.status_code)
        print("📄 응답본문:", response.text)

        if response.status_code == 200:
            return {
                "message": "등급 정보 전송 성공",
                "server_response": response.json()
            }
        else:
            return {
                "message": "전송 실패",
                "status_code": response.status_code,
                "detail": response.text
            }

    except Exception as e:
        print("❌ [FastAPI] 예외 발생:", str(e))
        return {
            "message": "Spring 서버 요청 중 예외 발생",
            "error": str(e)
        }

