"""검사 결과를 외부 Spring 서버로 중계하는 로컬 FastAPI 서버.

Jetson의 제어 코드(``controller.py``)는 등급 집계를 이 서버로 POST하고, 이 서버가
Spring에 로그인해 accessToken을 받은 뒤 검사 결과를 전달합니다. 토큰 발급을 제어
코드에서 떼어내기 위한 중계 계층입니다.

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from typing import Dict

import requests
from fastapi import FastAPI
from pydantic import BaseModel

import _bootstrap  # noqa: F401  (sys.path 설정)
import config

app = FastAPI(title="SugAI 검사 결과 중계")


class InspectionData(BaseModel):
    """제어 코드가 보내는 요청 본문. 예: ``{"fruitTypeId": 1, "counts": {"S": 30, ...}}``"""

    fruitTypeId: int
    counts: Dict[str, int]


def get_access_token():
    """Spring에 로그인해 accessToken을 받아옵니다. 실패하면 ``None``."""
    payload = {"username": config.SPRING_USERNAME, "password": config.SPRING_PASSWORD}

    try:
        res = requests.post(
            config.SPRING_LOGIN_URL, json=payload, timeout=config.HTTP_TIMEOUT
        )
        res.raise_for_status()
    except Exception as e:
        print("[API] Spring 로그인 실패:", e)
        return None

    # Spring의 CommonResponse 구조: {"data": {"accessToken": "..."}, ...}
    token = (res.json().get("data") or {}).get("accessToken")
    if not token:
        print("[API] 로그인 응답에 accessToken이 없습니다:", res.text[:200])
        return None
    return token


@app.post("/send-inspection")
def send_inspection(data: InspectionData):
    """등급 집계를 받아 Spring 검사 API로 전달합니다.

    전송이 실패해도 HTTP 200으로 응답하고 본문의 ``ok``로 성공 여부를 알립니다.
    제어 코드가 END 처리를 계속 진행할 수 있어야 하기 때문입니다.
    """
    print("[API] 집계 수신: fruitTypeId={}, counts={}".format(data.fruitTypeId, data.counts))

    token = get_access_token()
    if not token:
        return {"ok": False, "message": "accessToken 발급 실패"}

    payload = {"fruitTypeId": data.fruitTypeId, "counts": data.counts}
    headers = {"Authorization": "Bearer {}".format(token)}

    try:
        res = requests.post(
            config.SPRING_INSPECTION_URL,
            json=payload,
            headers=headers,
            timeout=config.HTTP_TIMEOUT,
        )
    except Exception as e:
        print("[API] Spring 전송 중 예외:", e)
        return {"ok": False, "message": "Spring 서버 요청 중 예외 발생", "error": str(e)}

    print("[API] Spring 응답 {}: {}".format(res.status_code, res.text[:200]))

    if res.status_code != 200:
        return {
            "ok": False,
            "message": "전송 실패",
            "status_code": res.status_code,
            "detail": res.text[:500],
        }

    try:
        server_response = res.json()
    except ValueError:
        server_response = res.text[:500]
    return {"ok": True, "message": "등급 정보 전송 성공", "server_response": server_response}
