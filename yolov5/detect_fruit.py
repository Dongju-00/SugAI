"""과일 검출 · 추적 · crop 저장 · 당도 추론.

카메라 프레임에서 YOLOv5로 과일을 찾고, SORT로 추적 ID를 붙여 **같은 과일을 한 번만**
저장합니다. crop을 저장한 즉시 당도 모델을 호출해 등급을 표준출력에 찍습니다.

    python detect_fruit.py --weights best.pt --view-img

제어 코드(``server/controller.py``)는 이 스크립트를 별도 프로세스로 띄우고 아래 줄을
읽어 등급 큐에 넣습니다. 형식을 바꾸면 ``controller.py``도 함께 고쳐야 합니다.

    CAPTURE:<경로>              crop 저장됨
    GRADE:<등급>:<경로>         당도 추론 결과
    MODEL_ERROR:<경로>:<사유>   추론 실패
    DETECT_DONE                 루프 종료

종료는 SIGTERM/Ctrl+C, 정지 신호 파일(``--stopfile``), 미리보기 창에서 q/ESC 중
무엇으로도 가능합니다.

YOLOv5 코드를 그대로 쓰기 때문에 이 파일은 ``yolov5/`` 안에 있어야 합니다.
"""

import argparse
import os
import signal
import sys
from pathlib import Path

# GStreamer 백엔드 우선순위는 cv2를 import하기 전에 정해야 반영됩니다.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "1")

import cv2
import numpy as np
import torch

# ===== YOLOv5 모듈 경로 =====
YOLO_DIR = Path(__file__).resolve().parent
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from models.experimental import attempt_load
from sort import Sort
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import check_file, check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# ===== 저장소 설정 · 당도 모델 =====
sys.path.insert(0, str(YOLO_DIR.parent))
import config  # noqa: E402

sys.path.insert(0, str(config.BRIX_MODEL_DIR))
from predictor import load_ensemble, predict_brix, brix_to_grade  # noqa: E402

#: 미리보기 박스 두께.
LINE_THICKNESS = 2
#: SORT 추적 파라미터. min_hits=2라 두 프레임 이상 잡힌 과일만 ID를 받습니다.
TRACKER_MAX_AGE = 15
TRACKER_MIN_HITS = 2
TRACKER_IOU_THRES = 0.3

#: SIGTERM/SIGINT을 받으면 True가 되고, 루프가 다음 프레임에서 빠져나옵니다.
_terminate = False


def _request_terminate(signum, frame):
    global _terminate
    _terminate = True


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _request_terminate)
    except (ValueError, OSError):
        pass  # 메인 스레드가 아니면 등록할 수 없습니다.


# ===============================
# 1) 입력 소스
# ===============================
def is_gstreamer_source(source):
    """GStreamer 파이프라인 문자열인지 판별합니다."""
    s = str(source).strip()
    return (
        "nvarguscamerasrc" in s
        or "appsink" in s
        or " !" in s
        or s.startswith("v4l2src ")
    )


def gstreamer_frames(pipeline, img_size, stride):
    """GStreamer 파이프라인에서 ``(모델 입력, 원본 프레임)``을 계속 내보냅니다."""
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("GStreamer 파이프라인 오픈 실패: {}".format(pipeline))
    try:
        while True:
            ok, im0 = cap.read()
            if not ok or im0 is None:
                continue  # 프레임 드롭은 흔하므로 그냥 다음 프레임을 기다립니다.
            img = letterbox(im0, new_shape=img_size, stride=stride, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            yield np.ascontiguousarray(img), im0
    finally:
        cap.release()


def yolo_frames(dataset):
    """YOLOv5 기본 로더의 5-튜플을 ``(모델 입력, 원본 프레임)``으로 맞춥니다."""
    for _path, img, im0s, _cap, _s in dataset:
        # LoadStreams는 카메라별 리스트를, LoadImages는 배열 하나를 줍니다.
        im0 = im0s if isinstance(im0s, np.ndarray) else im0s[0]
        yield img, im0


def open_source(source, img_size, stride):
    """소스 종류를 보고 알맞은 프레임 제너레이터를 엽니다."""
    source = str(source)

    if is_gstreamer_source(source):
        return gstreamer_frames(source, img_size, stride)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    if is_url and is_file:
        source = check_file(source)

    if source.isnumeric() or source.endswith(".txt") or (is_url and not is_file):
        return yolo_frames(LoadStreams(source, img_size=img_size, stride=stride))
    return yolo_frames(LoadImages(source, img_size=img_size, stride=stride))


# ===============================
# 2) 검출 루프
# ===============================
@torch.no_grad()
def run(
    weights,
    source,
    save_dir,
    stopfile,
    imgsz=(640, 640),
    conf_thres=0.6,
    iou_thres=0.45,
    max_det=50,
    device="",
    half=False,
    view_img=False,
    save_crops=True,
):
    """검출 루프를 끝까지 돌립니다."""
    if save_crops:
        load_ensemble()  # 무거운 초기화는 루프 전에 한 번만.

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stopfile = Path(stopfile)

    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    stride = int(getattr(model, "stride", torch.tensor([32])).max())
    imgsz = check_img_size(imgsz, s=stride)
    use_half = half and device.type != "cpu"
    if use_half:
        model.half()
    names = model.module.names if hasattr(model, "module") else model.names

    tracker = Sort(
        max_age=TRACKER_MAX_AGE,
        min_hits=TRACKER_MIN_HITS,
        iou_threshold=TRACKER_IOU_THRES,
    )
    #: 이미 저장한 추적 ID. 같은 과일을 두 번 저장하지 않기 위한 기록입니다.
    saved_track_ids = set()
    #: 저장 파일명 번호. 검사마다 1부터 시작하므로 캡처 폴더는 비워진 상태여야 합니다.
    next_file_number = 1

    def handle_crop(frame, box, track_id):
        """추적 ID를 처음 본 과일의 crop을 저장하고 등급을 출력합니다."""
        nonlocal next_file_number

        x1, y1, x2, y2 = [int(max(v, 0)) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        out_path = save_dir / "{}.jpg".format(next_file_number)
        cv2.imwrite(str(out_path), crop)
        print("CAPTURE:{}".format(out_path), flush=True)

        try:
            # 추론은 한 번만 돌리고 등급은 그 값에서 바로 계산합니다.
            brix = predict_brix(str(out_path))
            print("GRADE:{}:{}".format(brix_to_grade(brix), out_path), flush=True)
        except Exception as e:
            print("MODEL_ERROR:{}:{}".format(out_path, e), flush=True)

        saved_track_ids.add(track_id)
        next_file_number += 1

    def process_frame(img, frame):
        """한 프레임을 검출·추적·저장합니다. 계속 돌릴지 여부를 돌려줍니다."""
        im = torch.from_numpy(img).to(device)
        im = im.half() if use_half else im.float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im)[0]
        detections = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        annotator = Annotator(frame, line_width=LINE_THICKNESS, example=str(names))

        for det in detections:
            if not len(det):
                continue

            # 모델 입력 좌표를 원본 프레임 좌표로 되돌립니다.
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det.tolist():
                c = int(cls)
                annotator.box_label(
                    xyxy, "{} {:.2f}".format(names[c], conf), color=colors(c, True)
                )

            for x1, y1, x2, y2, track_id in tracker.update(det[:, :5].cpu().numpy()):
                track_id = int(track_id)
                annotator.box_label(
                    [x1, y1, x2, y2], "ID {}".format(track_id), color=(0, 255, 0)
                )
                if save_crops and track_id not in saved_track_ids:
                    handle_crop(frame, (x1, y1, x2, y2), track_id)

        if view_img:
            cv2.imshow("detect_fruit", annotator.result())
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                return False
        return True

    frames = open_source(source, imgsz, stride)
    try:
        for img, frame in frames:
            if _terminate or stopfile.exists():
                break
            if not process_frame(img, frame):
                break
    finally:
        frames.close()
        cv2.destroyAllWindows()

    print("DETECT_DONE", flush=True)


# ===============================
# 3) CLI
# ===============================
def parse_opt():
    parser = argparse.ArgumentParser(description="과일 검출 · 당도 추론")
    parser.add_argument("--weights", type=str, default=str(config.DETECT_WEIGHTS),
                        help="YOLOv5 검출 가중치")
    parser.add_argument("--source", type=str, default=config.CAMERA_SOURCE,
                        help="GStreamer 파이프라인, 카메라 번호, 이미지/영상 경로, URL")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int,
                        default=[config.DETECT_IMG_SIZE], help="추론 입력 크기")
    parser.add_argument("--conf-thres", type=float, default=config.DETECT_CONF_THRES)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=config.DETECT_MAX_DET)
    parser.add_argument("--device", default="", help="예: 0 또는 cpu")
    parser.add_argument("--half", action="store_true", help="FP16 추론 (GPU 전용)")
    parser.add_argument("--view-img", action="store_true", help="미리보기 창 표시")
    parser.add_argument("--save-dir", type=str, default=str(config.CAPTURES_DIR),
                        help="crop을 저장할 폴더")
    parser.add_argument("--stopfile", type=str, default=str(config.STOPFILE),
                        help="이 파일이 생기면 루프를 종료합니다")
    parser.add_argument("--no-save", action="store_true",
                        help="crop 저장과 당도 추론을 끄고 검출만 확인합니다")

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz = opt.imgsz * 2
    return opt


def main():
    opt = parse_opt()
    run(
        weights=opt.weights,
        source=opt.source,
        save_dir=opt.save_dir,
        stopfile=opt.stopfile,
        imgsz=tuple(opt.imgsz),
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        half=opt.half,
        view_img=opt.view_img,
        save_crops=not opt.no_save,
    )


if __name__ == "__main__":
    main()
