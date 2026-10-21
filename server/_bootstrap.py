"""저장소 루트를 ``sys.path``에 추가해 루트의 ``config.py``를 import할 수 있게 합니다.

server/ 안의 모듈은 패키지가 아니라 단독 스크립트로 실행되기 때문에(``python app.py``,
``uvicorn api:app``) 저장소 루트가 자동으로 ``sys.path``에 들어오지 않습니다.
그래서 ``import config`` 앞에 이 모듈을 먼저 import합니다.

    import _bootstrap  # noqa: F401  (sys.path 설정)
    import config
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
