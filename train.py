import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import utils

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg = utils.load_yaml(args.config)
classes = utils.load_classes(cfg["data"]["classes_yaml"])

# ----- 시드 고정 (교수님 코드 스타일) -----
if cfg["train"].get("cpu_only", True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 반드시 tensorflow import 전에 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random as rn
import numpy as np
import tensorflow as tf

seed_num = cfg["train"].get("seed", 0)
np.random.seed(seed_num)
rn.seed(seed_num)
tf.random.set_seed(seed_num)
# ---------------------------------------

from src import data as data_mod
from src import model as model_mod
from src import viz

run_dir = utils.make_run_dir("runs", cfg["exp_name"])
utils.save_config_copy(args.config, run_dir)
fig_dir = os.path.join(run_dir, "figures")
data_cfg = cfg["data"]

# 1) instance 목록 생성 (train/val/test 폴더 = 세션 분할 그대로, 재분할하지 않음)
splits = {}
for split in ["train", "val", "test"]:
    splits[split] = data_mod.build_instance_index(os.path.join(data_cfg["root"], split), classes, data_cfg["other"])
    print(f"[data] {split}: {len(splits[split])} instances")

counts_by_split = {s: data_mod.class_counts(splits[s], classes) for s in splits}
print("[data] 클래스별 샘플 수:", json.dumps(counts_by_split, ensure_ascii=False))
viz.plot_class_distribution(counts_by_split, os.path.join(fig_dir, "class_distribution.png"))

if len(splits["train"]) == 0:
    raise SystemExit("train instance가 0개입니다. datasets/tools/train/{images,labels} 를 채운 뒤 다시 실행하세요.")


def save_sample_grid(instances, split_name, per_class=4):
    by_class = {}
    for inst in instances:
        by_class.setdefault(inst["class_id"], []).append(inst)
    seq = data_mod.ToolCropSequence(instances, classes, data_cfg, augment_cfg=None, batch_size=1, train=False)
    imgs, titles = [], []
    for cls_id, name in classes["id2name"].items():
        for inst in by_class.get(cls_id, [])[:per_class]:
            x, _ = seq._build_one(inst)
            imgs.append(x)
            titles.append(name)
    viz.plot_sample_grid(imgs, titles, os.path.join(fig_dir, f"sample_grid_{split_name}.png"),
                          f"{split_name} 클래스별 crop 예시", n_cols=per_class)


# 2) 샘플 그리드 + 증강 전/후 비교
save_sample_grid(splits["train"], "train")
if len(splits["val"]) > 0:
    save_sample_grid(splits["val"], "val")

if cfg["augment"].get("enable", True):
    tool_insts = [i for i in splits["train"] if not i["is_other"]][:6] or splits["train"][:6]
    seq_plain = data_mod.ToolCropSequence(tool_insts, classes, data_cfg, augment_cfg=None, batch_size=1, train=False)
    seq_aug = data_mod.ToolCropSequence(tool_insts, classes, data_cfg, augment_cfg=cfg["augment"], batch_size=1,
                                         train=True, seed=seed_num)
    before = [seq_plain._build_one(inst)[0] for inst in tool_insts]
    after = [seq_aug._build_one(inst)[0] for inst in tool_insts]
    viz.plot_augmentation_comparison(before, after, os.path.join(fig_dir, "augmentation_before_after.png"),
                                      n_show=len(tool_insts))

# 3) Sequence 생성
train_seq = data_mod.ToolCropSequence(splits["train"], classes, data_cfg, augment_cfg=cfg["augment"],
                                       batch_size=cfg["train"]["batch_size"], train=True, seed=seed_num)
val_seq = None
if len(splits["val"]) > 0:
    val_seq = data_mod.ToolCropSequence(splits["val"], classes, data_cfg, augment_cfg=None,
                                         batch_size=cfg["train"]["batch_size"], train=False, seed=seed_num)

# 4) 모델 생성
input_shape = (data_cfg["input_size"], data_cfg["input_size"], data_mod.input_channels(data_cfg["input_mode"]))
if cfg["model"]["type"] == "mobilenet_v2" and data_cfg["input_mode"] == "rgbm":
    raise SystemExit("mobilenet_v2는 input_mode=rgbm(4채널)을 지원하지 않습니다. rgb/masked_rgb를 사용하세요.")

model = model_mod.build_model(cfg["model"], input_shape, classes["num_classes"])
model_mod.compile_model(model, cfg["train"])
model.summary()

viz.save_model_summary(model, os.path.join(run_dir, "model_summary.txt"))
viz.save_layer_table(model, os.path.join(run_dir, "layer_table.csv"))
viz.try_plot_model_diagram(model, os.path.join(fig_dir, "model_diagram.png"))

# 5) 콜백 + 학습 실행
ckpt_path = os.path.join(run_dir, "best.h5")
monitor_metric = cfg["train"].get("monitor", "val_accuracy") if val_seq is not None else "accuracy"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor=monitor_metric, save_best_only=True, mode="max",
                                        verbose=1),
    tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
]
patience = cfg["train"].get("early_stopping_patience")
if patience and val_seq is not None:
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, mode="max", patience=patience,
                                                        restore_best_weights=True))

history = model.fit(train_seq, validation_data=val_seq, epochs=cfg["train"]["epochs"], callbacks=callbacks,
                     verbose=1)

utils.save_json(history.history, os.path.join(run_dir, "history.json"))
viz.plot_training_curves(history.history, os.path.join(fig_dir, "training_curves.png"))
if not os.path.exists(ckpt_path):
    model.save(ckpt_path)  # val_seq가 없을 때의 안전장치

utils.append_summary_csv("runs", {
    "exp_name": cfg["exp_name"], "model_type": cfg["model"]["type"], "input_mode": data_cfg["input_mode"],
    "input_size": data_cfg["input_size"], "augment": cfg["augment"].get("enable", True),
    "epochs_ran": len(history.history["loss"]),
    "final_train_acc": history.history.get("accuracy", [None])[-1],
    "final_val_acc": history.history.get("val_accuracy", [None])[-1],
    "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]), "ckpt": ckpt_path,
})
print(f"[완료] 결과: {run_dir}")
