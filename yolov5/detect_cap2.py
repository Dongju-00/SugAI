# detect_cap.py
# Jetson Nano + YOLOv5 v6.x 호환 / GStreamer 파이프라인 대응 버전 (argparse 버그 수정: store_true)

import argparse
import os
import sys
from pathlib import Path
from typing import Generator, Tuple, Any

import cv2
import numpy as np
import torch

# ===== YOLOv5 환경 경로 세팅 =====
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 환경 (GStreamer 우선)
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "1")

# ===== YOLOv5 유틸 =====
from sort import Sort  # sort.py 또는 pip로 준비
from utils.augmentations import letterbox
from utils.general import (
    LOGGER, check_file, check_img_size, non_max_suppression, scale_coords, increment_path
)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams


def is_gstreamer_source(src: str) -> bool:
    """GStreamer 파이프라인 문자열인지 검사."""
    if not isinstance(src, str):
        return False
    s = src.strip()
    return ("nvarguscamerasrc" in s) or ("appsink" in s) or (" !" in s) or s.startswith("v4l2src ") or s.endswith(" appsink")


def gst_stream_loader(pipeline: str, img_size: Tuple[int, int], stride: int) -> Generator[Tuple[Any, np.ndarray, np.ndarray, Any, str], None, None]:
    """
    GStreamer 파이프라인을 직접 열어 YOLOv5가 기대하는 배치를 생성.
    yield 형식: (path, img, im0, cap, s)
      - img: (3, H, W) RGB, letterbox 적용, contiguous ndarray
      - im0: 원본 BGR 프레임
    """
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise Exception(
            "GStreamer 파이프라인을 열 수 없습니다. 아래 확인:\n"
            "1) OpenCV가 GStreamer로 빌드되었는지\n"
            "2) 파이프라인 철자/공백/느낌표/쉼표\n"
            "3) nvarguscamerasrc 카메라 권한/연결 상태\n"
            f"파이프라인: {pipeline}"
        )

    while True:
        ok, im0 = cap.read()
        if not ok or im0 is None:
            continue

        img = letterbox(im0, new_shape=img_size, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        s = ''
        yield pipeline, img, im0, cap, s


@torch.no_grad()
def run(
    weights=ROOT / 'yolov5s.pt',
    source=ROOT / 'data/images',    # 파일/폴더/URL/또는 GStreamer 문자열
    imgsz=(640, 640),
    conf_thres=0.6,
    iou_thres=0.45,
    max_det=1000,
    device='',
    view_img=False,                 # ← 이 플래그를 빼고 실행하면 완전 headless (GUI 없음)
    save_txt=False,
    save_conf=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    project=ROOT / 'runs/detect',
    name='exp',
    exist_ok=False,
    line_thickness=2,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    save_dir_override=None,  # 외부 저장 디렉터리 강제
    save_path='',            # 정확한 파일 경로(한 장) 저장
    save_crop_once=True,     # 같은 트랙 ID는 한 번만 크롭 저장
):
    source = str(source)
    save_img = not nosave

    # ====== 입력 소스 유형 판별 ======
    gst_mode = is_gstreamer_source(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    if is_url and is_file:
        source = check_file(source)  # download

    webcam = (not gst_mode) and (source.isnumeric() or source.endswith('.txt') or (is_url and not is_file))

    # ====== 저장 경로 준비 ======
    if save_dir_override:
        save_dir = Path(save_dir_override)
        save_dir.mkdir(parents=True, exist_ok=True)
        project = save_dir.parent
        name = save_dir.name
        exist_ok = True

    # ====== 모델/장치 ======
    device = select_device(device)
    model = attempt_load(weights, map_location=device)  # FP32
    stride = int(getattr(model, 'stride', torch.tensor([32])).max())
    imgsz = check_img_size(imgsz, s=stride)
    if half and device.type != 'cpu':
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names

    # ====== 데이터 로더 준비 ======
    if gst_mode:
        dataset = gst_stream_loader(source, imgsz, stride)
        use_custom_iter = True
    elif webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)  # vid_stride 미사용(구버전 호환)
        use_custom_iter = False
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        use_custom_iter = False

    # ====== 실행 ======
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)
    saved_ids = set()
    img_counter = 1

    def inference_step(im_tensor, im0):
        nonlocal img_counter
        pred = model(im_tensor, augment=augment)[0]
        dets = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        for det in dets:
            if len(det):
                det[:, :4] = scale_coords(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                # YOLO 라벨
                for *xyxy, conf, cls in det.tolist():
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))

                # SORT 추적 (xyxy + conf)
                det_np = det[:, :5].cpu().numpy()
                tracks = tracker.update(det_np)  # [x1,y1,x2,y2,track_id]

                # 트랙 박스 + ID 표시 및 1회 크롭 저장
                for x1, y1, x2, y2, tid in tracks:
                    tid = int(tid)
                    annotator.box_label([x1, y1, x2, y2], f"ID {tid}", color=(0, 255, 0))
                    if save_crop_once and (tid not in saved_ids):
                        x1i, y1i, x2i, y2i = map(lambda v: int(max(v, 0)), [x1, y1, x2, y2])
                        crop = im0[y1i:y2i, x1i:x2i]
                        if crop.size != 0 and save_img:
                            if save_path:
                                out_path = Path(save_path)
                            else:
                                out_path = save_dir / f"{img_counter}.jpg"
                            cv2.imwrite(str(out_path), crop)
                            LOGGER.info(f"💾 Saved: {out_path}")
                            saved_ids.add(tid)
                            img_counter += 1

        vis = annotator.result()
        # ==== ESC / q 종료 처리 (GUI가 켜진 경우에만) ====
        if view_img:
            cv2.imshow("detect_cap", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):   # 27 = ESC, 또는 'q'
                return False
        return True

    # ====== 루프 ======
    try:
        if use_custom_iter:
            for path, img, im0, cap, s in dataset:
                im = torch.from_numpy(img).to(device)
                im = im.half() if (half and device.type != 'cpu') else im.float()
                im /= 255.0
                if im.ndimension() == 3:
                    im = im.unsqueeze(0)

                cont = inference_step(im, im0)
                if not cont:
                    break
        else:
            for path, im, im0s, vid_cap, s in dataset:
                im_tensor = torch.from_numpy(im).to(device)
                im_tensor = im_tensor.half() if (half and device.type != 'cpu') else im_tensor.float()
                im_tensor /= 255.0
                if im_tensor.ndimension() == 3:
                    im_tensor = im_tensor.unsqueeze(0)

                im0 = im0s if isinstance(im0s, np.ndarray) else im0s[0]
                cont = inference_step(im_tensor, im0)
                if not cont:
                    break
    finally:
        # 리소스 정리
        try:
            if use_custom_iter:
                cap.release()  # gst_stream_loader에서의 cap
        except Exception:
            pass
        cv2.destroyAllWindows()

    LOGGER.info(f"✅ Done. Results saved to {save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(ROOT / 'yolov5s.pt'), help='model path')
    parser.add_argument('--source', type=str, default=str(ROOT / 'data/images'), help='file/dir/URL or GStreamer pipeline')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results (창 표시). 미지정 시 GUI 없음')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0 or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=str(ROOT / 'runs/detect'), help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--save-dir', type=str, default='', help='외부 저장 디렉터리 (예: /home/jetson/brixproject/captures)')
    parser.add_argument('--save-path', type=str, default='', help='정확한 파일 경로(한 장)로 저장')
    parser.add_argument('--save-crop-once', action='store_true', help='같은 트랙 ID는 한 번만 크롭 저장')
    opt = parser.parse_args()

    # imgsz parsing
    opt.imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz
    return opt


def main(opt):
    run(
        weights=opt.weights,
        source=opt.source,
        imgsz=tuple(opt.imgsz),
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        view_img=opt.view_img,
        save_txt=opt.save_txt,
        save_conf=getattr(opt, 'save_conf', False),
        nosave=opt.nosave,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
        augment=opt.augment,
        visualize=opt.visualize,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        line_thickness=opt.line_thickness,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        half=opt.half,
        dnn=opt.dnn,
        save_dir_override=opt.save_dir if opt.save_dir else None,
        save_path=opt.save_path,
        save_crop_once=(opt.save_crop_once or True),
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

