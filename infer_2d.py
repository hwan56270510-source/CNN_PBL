import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import utils

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="이미지 폴더 경로 (또는 이미지 1장)")
parser.add_argument("--yolo-weights", default=None, help="YOLO-Seg .pt 경로 (이 PC에서 직접 YOLO를 돌릴 때)")
parser.add_argument("--yolo-json-dir", default=None,
                     help="yolo_export.py 로 미리 뽑아둔 JSON 폴더 (YOLO 담당자가 결과만 넘겨줄 때, ultralytics 불필요)")
parser.add_argument("--config", default="configs/cnn.yaml")
parser.add_argument("--ckpt", required=True, help="CNN 체크포인트(.h5) 경로")
parser.add_argument("--out", default="runs/demo_infer_2d")
parser.add_argument("--conf", type=float, default=None, help="YOLO conf threshold override (--yolo-weights 사용 시)")
args = parser.parse_args()

if bool(args.yolo_weights) == bool(args.yolo_json_dir):
    raise SystemExit("--yolo-weights 또는 --yolo-json-dir 둘 중 하나만 지정하세요.")

yolo_model = None
if args.yolo_weights:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics가 설치되어 있지 않습니다. environment.md 참고.") from e
    yolo_model = YOLO(args.yolo_weights)

cfg = utils.load_yaml(args.config)
classes = utils.load_classes(cfg["data"]["classes_yaml"])

if cfg["train"].get("cpu_only", True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from src import data as data_mod

data_cfg = cfg["data"]
conf_thr = args.conf if args.conf is not None else cfg["eval"].get("yolo_conf_threshold", 0.25)
id2name = classes["id2name"]

# 클래스별 오버레이 색상 (BGR)
CLASS_COLORS = {"drill": (0, 255, 0), "hammer": (0, 165, 255), "clamp": (255, 0, 0), "other": (128, 128, 128)}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

cnn_model = tf.keras.models.load_model(args.ckpt)

out_overlay_dir = os.path.join(args.out, "overlays")
out_json_dir = os.path.join(args.out, "results")
os.makedirs(out_overlay_dir, exist_ok=True)
os.makedirs(out_json_dir, exist_ok=True)


def get_detections(img_path):
    # 반환: [(box_xyxy, yolo_conf, mask_polygon_or_None), ...]
    if yolo_model is not None:
        result = yolo_model.predict(source=img_path, conf=conf_thr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        polys = result.masks.xy if result.masks is not None else [None] * len(boxes)
        return list(zip(boxes, confs, polys))

    # --yolo-json-dir 모드: yolo_export.py가 저장한 JSON을 읽음
    stem = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(args.yolo_json_dir, f"{stem}.json")
    if not os.path.exists(json_path):
        print(f"  [경고] {json_path} 없음, 건너뜀")
        return []
    record = utils.load_json(json_path)
    return [(np.array(d["bbox_xyxy"], dtype=np.float32), d["yolo_conf"],
              np.array(d["mask_polygon"], dtype=np.float32) if d["mask_polygon"] else None)
             for d in record["detections"]]


def draw_detection(overlay, box, poly, pred_class, pred_conf, yolo_conf):
    color = CLASS_COLORS.get(pred_class, (255, 255, 255))
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    if poly is not None and len(poly) >= 3:
        cv2.polylines(overlay, [np.array(poly, dtype=np.int32).reshape(-1, 1, 2)], True, color, 2)
    label = f"{pred_class} {pred_conf:.2f} (yolo {yolo_conf:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(overlay, label, (x1 + 2, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def process_one_image(img_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    margin, size, mode = data_cfg["bbox_margin"], data_cfg["input_size"], data_cfg["input_mode"]
    detections_raw = get_detections(img_path)
    detections = []

    crops, valid_polys, valid_boxes, valid_confs = [], [], [], []
    for box, conf, poly in detections_raw:
        mask = data_mod.polygon_to_mask(poly, img_h, img_w) if (poly is not None and len(poly) >= 3) else None
        bbox = data_mod.expand_bbox(np.array(box, dtype=np.float32), margin, img_w, img_h)
        crop_img, crop_mask = data_mod.crop_and_letterbox(img_rgb, mask, bbox, size)
        if crop_img is None:
            continue
        crops.append(data_mod.build_input_array(crop_img, crop_mask, mode))
        valid_polys.append(poly)
        valid_boxes.append(box)
        valid_confs.append(conf)

    if crops:
        probs = cnn_model.predict(np.stack(crops, axis=0).astype(np.float32), verbose=0)
        for det_id, (box, conf, poly, p) in enumerate(zip(valid_boxes, valid_confs, valid_polys, probs)):
            pred_id = int(np.argmax(p))
            pred_class, pred_conf = id2name[pred_id], float(p[pred_id])
            detections.append({
                "det_id": det_id,
                "bbox_xyxy": [float(v) for v in box],
                "mask_polygon": [[float(x), float(y)] for x, y in poly] if poly is not None else [],
                "yolo_conf": float(conf),
                "cls_probs": {id2name[i]: float(p[i]) for i in range(len(p))},
                "pred_class": pred_class,
                "pred_conf": pred_conf,
            })
            draw_detection(overlay, box, poly, pred_class, pred_conf, conf)

    cv2.imwrite(os.path.join(out_overlay_dir, os.path.basename(img_path)), overlay)
    return {"image": os.path.basename(img_path), "image_wh": [img_w, img_h], "detections": detections}


if os.path.isdir(args.source):
    image_paths = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
                   if f.lower().endswith(IMG_EXTS)]
else:
    image_paths = [args.source]
print(f"[infer_2d] 처리할 이미지 {len(image_paths)}장")

with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as jsonl_f:
    for img_path in image_paths:
        record = process_one_image(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        utils.save_json(record, os.path.join(out_json_dir, f"{stem}.json"))
        jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {os.path.basename(img_path)}: {len(record['detections'])}개 검출")

print(f"[완료] 결과: {args.out}")
