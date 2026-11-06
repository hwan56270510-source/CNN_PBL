"""YOLO 담당자가 자신의 ultralytics 환경에서 실행하는 스크립트. TensorFlow 없이 동작함.
이미지 폴더 -> YOLO-Seg 추론 -> 박스/마스크만 JSON으로 저장 (CNN 분류는 하지 않음).
저장된 JSON 폴더를 infer_2d.py --yolo-json-dir 로 넘기면 CNN 쪽에서 이어받아 처리한다.

사용법:
    python yolo_export.py --source image_folder --weights best.pt --out yolo_results
"""
import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--out", default="yolo_results")
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
model = YOLO(args.weights)

if os.path.isdir(args.source):
    image_paths = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
                   if f.lower().endswith(IMG_EXTS)]
else:
    image_paths = [args.source]

for img_path in image_paths:
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    result = model.predict(source=img_path, conf=args.conf, verbose=False)[0]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        polys = result.masks.xy if result.masks is not None else [None] * len(boxes)
        for box, conf, poly in zip(boxes, confs, polys):
            detections.append({
                "bbox_xyxy": [float(v) for v in box],
                "mask_polygon": [[float(x), float(y)] for x, y in poly] if poly is not None else [],
                "yolo_conf": float(conf),
            })

    record = {"image": os.path.basename(img_path), "image_wh": [img_w, img_h], "detections": detections}
    stem = os.path.splitext(os.path.basename(img_path))[0]
    with open(os.path.join(args.out, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"{os.path.basename(img_path)}: {len(detections)}개 검출")

print(f"[완료] {args.out} 에 저장. 이 폴더를 infer_2d.py --yolo-json-dir 로 넘기세요.")
