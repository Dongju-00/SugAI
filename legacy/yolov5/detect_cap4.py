import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

# ===== YOLOv5 경로 =====
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "1")

# YOLOv5 유틸
from sort import Sort
from utils.augmentations import letterbox
from utils.general import LOGGER, check_file, check_img_size, non_max_suppression, scale_coords, increment_path
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams

# ===== 종료 플래그 (SIGTERM + SIGINT[Ctrl+C]) =====
_TERMINATE = False
def _handle_term(signum, frame):
    global _TERMINATE
    _TERMINATE = True
try:
    import signal
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT,  _handle_term)  # Ctrl+C 핸들링 추가
except Exception:
    pass

def is_gstreamer_source(src: str) -> bool:
    if not isinstance(src, str):
        return False
    s = src.strip()
    return ("nvarguscamerasrc" in s) or ("appsink" in s) or (" !" in s) or s.startswith("v4l2src ") or s.endswith(" appsink")

def gst_stream_loader(pipeline: str, img_size: Tuple[int, int], stride: int):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise Exception(f"GStreamer 파이프라인 오픈 실패: {pipeline}")
    while True:
        ok, im0 = cap.read()
        if not ok or im0 is None:
            continue
        img = letterbox(im0, new_shape=img_size, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        s = ''
        yield pipeline, img, im0, cap, s

@torch.no_grad()
def run(
    weights=ROOT / 'yolov5s.pt',
    source=ROOT / 'data/images',
    imgsz=(640, 640),
    conf_thres=0.6,
    iou_thres=0.45,
    max_det=1000,
    device='',
    view_img=False,
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
    save_dir_override=str(Path.home() / "jetson/brixproject/captures"),
    save_path='',
    save_crop_once=True,
    stopfile=str(Path.home() / "jetson/brixproject/stop_detect"),
):
    source = str(source)
    save_img = not nosave

    gst_mode = is_gstreamer_source(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)
    webcam = (not gst_mode) and (source.isnumeric() or source.endswith('.txt') or (is_url and not is_file))

    # 저장 경로
    if save_dir_override:
        save_dir = Path(save_dir_override)
        save_dir.mkdir(parents=True, exist_ok=True)
        project = save_dir.parent
        name = save_dir.name
        exist_ok = True

    # YOLO
    device_t = select_device(device)
    model = attempt_load(weights, map_location=device_t)
    stride = int(getattr(model, 'stride', torch.tensor([32])).max())
    imgsz = check_img_size(imgsz, s=stride)
    if half and device_t.type != 'cpu':
        model.half()
    names = model.module.names if hasattr(model, 'module') else model.names

    # 데이터 로더
    if gst_mode:
        dataset = gst_stream_loader(source, imgsz, stride)
        use_custom_iter = True
    elif webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        use_custom_iter = False
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        use_custom_iter = False

    # 실행
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)
    saved_ids = set()
    img_counter = 1

    # ===== 수동 시퀀스 등급 =====
    manual_grades = ["B", "S", "A", "C"]  # 원하는 순서
    manual_idx = 0

    def inference_step(im_tensor, im0):
        nonlocal img_counter, manual_idx
        pred = model(im_tensor, augment=augment)[0]
        dets = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        for det in dets:
            if len(det):
                det[:, :4] = scale_coords(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det.tolist():
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))

                det_np = det[:, :5].cpu().numpy()
                tracks = tracker.update(det_np)

                for x1, y1, x2, y2, tid in tracks:
                    tid = int(tid)
                    annotator.box_label([x1, y1, x2, y2], f"ID {tid}", color=(0, 255, 0))
                    if save_crop_once and (tid not in saved_ids):
                        x1i, y1i, x2i, y2i = map(lambda v: int(max(v, 0)), [x1, y1, x2, y2])
                        crop = im0[y1i:y2i, x1i:x2i]
                        if crop.size != 0 and save_img:
                            out_path = Path(save_path) if save_path else (save_dir / f"{img_counter}.jpg")
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(out_path), crop)
                            print(f"CAPTURE:{str(out_path)}", flush=True)

                            # ✅ 모델 대신 수동 등급 지정
                            grade = manual_grades[manual_idx % len(manual_grades)]
                            manual_idx += 1
                            print(f"GRADE:{grade}:{str(out_path)}", flush=True)

                            saved_ids.add(tid)
                            img_counter += 1

        vis = annotator.result()
        if view_img:
            cv2.imshow("detect2", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                return False
        return True

    try:
        if use_custom_iter:
            for path, img, im0, cap, s in dataset:
                if _TERMINATE or (stopfile and os.path.exists(stopfile)):
                    break
                im = torch.from_numpy(img).to(device_t)
                im = im.half() if (half and device_t.type != 'cpu') else im.float()
                im /= 255.0
                if im.ndimension() == 3:
                    im = im.unsqueeze(0)
                cont = inference_step(im, im0)
                if not cont:
                    break
        else:
            for path, im, im0s, vid_cap, s in dataset:
                if _TERMINATE or (stopfile and os.path.exists(stopfile)):
                    break
                im_tensor = torch.from_numpy(im).to(device_t)
                im_tensor = im_tensor.half() if (half and device_t.type != 'cpu') else im_tensor.float()
                im_tensor /= 255.0
                if im_tensor.ndimension() == 3:
                    im_tensor = im_tensor.unsqueeze(0)
                im0 = im0s if isinstance(im0s, np.ndarray) else im0s[0]
                cont = inference_step(im_tensor, im0)
                if not cont:
                    break
    finally:
        try:
            if 'cap' in locals():
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    print("DETECT_DONE", flush=True)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(ROOT / 'yolov5s.pt'))
    parser.add_argument('--source', type=str, default=str(ROOT / 'data/images'))
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--project', default=str(ROOT / 'runs/detect'))
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=2, type=int)
    parser.add_argument('--hide-labels', default=False, action='store_true')
    parser.add_argument('--hide-conf', default=False, action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--save-dir', type=str, default=str(Path.home() / "jetson/brixproject/captures"))
    parser.add_argument('--save-path', type=str, default='')
    parser.add_argument('--save-crop-once', action='store_true')
    parser.add_argument('--stopfile', type=str, default=str(Path.home() / "jetson/brixproject/stop_detect"))
    opt = parser.parse_args()
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
        save_dir_override=opt.save_dir,
        save_path=opt.save_path,
        save_crop_once=opt.save_crop_once,
        stopfile=opt.stopfile,
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
