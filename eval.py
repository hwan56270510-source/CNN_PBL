import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import utils

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--yolo-weights", default=None, help="지정 시 config의 eval.yolo_weights를 덮어씀")
args = parser.parse_args()

cfg = utils.load_yaml(args.config)
classes = utils.load_classes(cfg["data"]["classes_yaml"])

# ----- 시드 고정 (교수님 코드 스타일) -----
if cfg["train"].get("cpu_only", True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random as rn
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

seed_num = cfg["train"].get("seed", 0)
np.random.seed(seed_num)
rn.seed(seed_num)
tf.random.set_seed(seed_num)
# ---------------------------------------

from src import data as data_mod
from src import viz

run_dir = utils.make_run_dir("runs", cfg["exp_name"])
fig_dir = os.path.join(run_dir, "figures")
data_cfg = cfg["data"]
class_names = [classes["id2name"][i] for i in range(classes["num_classes"])]

model = tf.keras.models.load_model(args.ckpt)
print(f"[eval] 체크포인트 로드: {args.ckpt}")


def evaluate_and_plot(X, Y, metas, split, tag, grid_samples=16, extra_cols=None, extra_rows=None):
    probs = model.predict(X, verbose=0)
    pred_ids = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    true_ids = Y.argmax(axis=1)

    acc = float(accuracy_score(true_ids, pred_ids))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=range(len(class_names)), zero_division=0)
    cm = confusion_matrix(true_ids, pred_ids, labels=range(len(class_names)))
    viz.plot_confusion_matrix(cm, class_names, os.path.join(fig_dir, f"confusion_matrix_{split}_{tag}.png"))

    correct_idx = np.where(pred_ids == true_ids)[0]
    wrong_idx = np.where(pred_ids != true_ids)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(correct_idx)
    rng.shuffle(wrong_idx)

    def grid(idxs, title, path):
        idxs = idxs[:grid_samples]
        viz.plot_prediction_grid([X[i] for i in idxs], [class_names[true_ids[i]] for i in idxs],
                                  [class_names[pred_ids[i]] for i in idxs], [pred_conf[i] for i in idxs],
                                  path, title, n_show=len(idxs))

    grid(correct_idx, f"맞은 예측 예시 ({split})", os.path.join(fig_dir, f"pred_correct_{split}_{tag}.png"))
    grid(wrong_idx, f"틀린 예측 예시 ({split})", os.path.join(fig_dir, f"pred_wrong_{split}_{tag}.png"))
    save_misclassified_csv(wrong_idx, metas, true_ids, pred_ids, pred_conf,
                            os.path.join(run_dir, f"misclassified_{split}_{tag}.csv"), extra_cols, extra_rows)

    return {
        "split": split, "tag": tag, "n_samples": int(len(true_ids)), "accuracy": acc,
        "macro_f1": float(np.mean(f1)),
        "per_class": {class_names[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                        "f1": float(f1[i]), "support": int(support[i])}
                      for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
    }


def save_misclassified_csv(wrong_idx, metas, true_ids, pred_ids, pred_conf, csv_path, extra_cols, extra_rows):
    import csv

    extra_cols = extra_cols or []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class", "pred_class", "pred_conf", *extra_cols, "원인_메모"])
        for i in wrong_idx:
            extra = extra_rows[i] if extra_rows else [""] * len(extra_cols)
            writer.writerow([metas[i]["img_path"], class_names[true_ids[i]], class_names[pred_ids[i]],
                              f"{pred_conf[i]:.3f}", *extra, ""])


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_e2e_eval(yolo_weights, split_dir, eval_cfg, split):
    # YOLO 예측 박스/마스크로 crop -> CNN 평가 (실전 성능). GT와 IoU 매칭 안 되면 other로 간주
    # (= YOLO 오검출을 CNN이 other로 걸러내는지 확인하는 부분)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics가 설치되어 있지 않습니다. environment.md 참고.") from e

    yolo = YOLO(yolo_weights)
    conf_thr = eval_cfg.get("yolo_conf_threshold", 0.25)
    iou_thr = eval_cfg.get("yolo_iou_match_threshold", 0.3)
    margin, size, mode = data_cfg["bbox_margin"], data_cfg["input_size"], data_cfg["input_mode"]
    other_id = classes["other_class_id"]

    X_list, true_ids, metas, yolo_confs, gt_matched = [], [], [], [], []
    for img_path, lbl_path in data_mod.list_split_files(split_dir):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]
        gt_objs = data_mod.parse_label_file(lbl_path, img_w, img_h)

        result = yolo.predict(source=img_path, conf=conf_thr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        polys = result.masks.xy if result.masks is not None else [None] * len(boxes)

        for box, conf, poly in zip(boxes, confs, polys):
            best_iou, best_obj = 0.0, None
            for obj in gt_objs:
                iou = bbox_iou(box, obj["bbox"])
                if iou > best_iou:
                    best_iou, best_obj = iou, obj
            matched = best_obj is not None and best_iou >= iou_thr
            gt_class_id = best_obj["class_id"] if matched else other_id

            mask = data_mod.polygon_to_mask(np.array(poly, dtype=np.float32), img_h, img_w) if (
                poly is not None and len(poly) >= 3) else None
            bbox = data_mod.expand_bbox(np.array(box, dtype=np.float32), margin, img_w, img_h)
            crop_img, crop_mask = data_mod.crop_and_letterbox(img_rgb, mask, bbox, size)
            if crop_img is None:
                continue

            X_list.append(data_mod.build_input_array(crop_img, crop_mask, mode))
            true_ids.append(gt_class_id)
            metas.append({"img_path": img_path, "class_id": gt_class_id, "is_other": gt_class_id == other_id})
            yolo_confs.append(float(conf))
            gt_matched.append(matched)

    if not X_list:
        print("[e2e] YOLO가 아무것도 검출하지 못했습니다 (conf_threshold를 낮춰보세요).")
        return {"split": split, "tag": "e2e", "n_samples": 0, "accuracy": None, "macro_f1": None, "per_class": {}}

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = tf.keras.utils.to_categorical(true_ids, num_classes=classes["num_classes"])
    extra_cols = ["yolo_conf", "yolo_gt_matched"]
    extra_rows = [[f"{yolo_confs[i]:.3f}", "매칭됨" if gt_matched[i] else "GT와_매칭안됨(오검출_가능성)"]
                  for i in range(len(metas))]
    return evaluate_and_plot(X, Y, metas, split, "e2e", eval_cfg.get("grid_samples", 16), extra_cols, extra_rows)


# ----- GT crop 평가 -----
split_dir = os.path.join(data_cfg["root"], args.split)
instances = data_mod.build_instance_index(split_dir, classes, data_cfg["other"])
print(f"[eval] {args.split} instance 수: {len(instances)}")
if len(instances) == 0:
    raise SystemExit("평가할 instance가 없습니다. 데이터셋을 채운 뒤 다시 실행하세요.")

X, Y, metas = data_mod.build_dataset_arrays(instances, classes, data_cfg)
metrics_gt = evaluate_and_plot(X, Y, metas, args.split, "gt", cfg["eval"].get("grid_samples", 16))
utils.save_json(metrics_gt, os.path.join(run_dir, f"metrics_gt_{args.split}.json"))

# ----- 필터 / 특징맵 시각화 (deep_cnn_filterVis.py / deep_cnn_featureMapVis.py 방식) -----
conv_names = [l.name for l in model.layers if isinstance(l, layers.Conv2D)]
if conv_names:
    viz.plot_conv_filters(model, conv_names[0], os.path.join(fig_dir, f"filters_{conv_names[0]}.png"))
    if len(conv_names) > 1:
        mid_name = conv_names[len(conv_names) // 2]
        viz.plot_conv_filters(model, mid_name, os.path.join(fig_dir, f"filters_{mid_name}.png"))
    sample_idx = min(cfg["eval"].get("featuremap_sample_index", 0), len(X) - 1)
    viz.plot_feature_maps(model, conv_names, X[sample_idx:sample_idx + 1], os.path.join(fig_dir, "feature_maps.png"))

# ----- YOLO 예측 crop 기반 end-to-end 평가 (선택) -----
yolo_weights = args.yolo_weights or cfg["eval"].get("yolo_weights")
if yolo_weights:
    metrics_e2e = run_e2e_eval(yolo_weights, split_dir, cfg["eval"], args.split)
    utils.save_json(metrics_e2e, os.path.join(run_dir, f"metrics_e2e_{args.split}.json"))
    utils.append_summary_csv("runs", {
        "exp_name": cfg["exp_name"], "stage": "eval_e2e", "split": args.split,
        "model_type": cfg["model"]["type"], "input_mode": data_cfg["input_mode"],
        "accuracy": metrics_e2e["accuracy"], "macro_f1": metrics_e2e["macro_f1"],
        "n_samples": metrics_e2e["n_samples"], "ckpt": args.ckpt, "yolo_weights": yolo_weights,
    })

utils.append_summary_csv("runs", {
    "exp_name": cfg["exp_name"], "stage": "eval_gt", "split": args.split,
    "model_type": cfg["model"]["type"], "input_mode": data_cfg["input_mode"],
    "accuracy": metrics_gt["accuracy"], "macro_f1": metrics_gt["macro_f1"],
    "n_samples": metrics_gt["n_samples"], "ckpt": args.ckpt,
})
print(f"[완료] {args.split} 평가 결과: {run_dir}")
