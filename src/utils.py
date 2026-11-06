import os
import csv
import json
import shutil
from datetime import datetime

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(classes_yaml_path):
    cfg = load_yaml(classes_yaml_path)
    id2name = {int(k): v for k, v in cfg["names"].items()}
    name2id = {v: k for k, v in id2name.items()}
    return {
        "id2name": id2name,
        "name2id": name2id,
        "num_classes": int(cfg["num_classes"]),
        "other_class_id": int(cfg["other_class_id"]),
    }


def make_run_dir(runs_root, exp_name):
    run_dir = os.path.join(runs_root, exp_name)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    return run_dir


def save_config_copy(config_path, run_dir):
    shutil.copy(config_path, os.path.join(run_dir, "config_used.yaml"))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_summary_csv(runs_root, row: dict):
    # 실험 1회 결과를 runs/summary.csv 에 한 줄 추가. 컬럼은 합집합으로 갱신.
    csv_path = os.path.join(runs_root, "summary.csv")
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}

    existing_rows = []
    fieldnames = list(row.keys())
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            for fn in reader.fieldnames or []:
                if fn not in fieldnames:
                    fieldnames.append(fn)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(row)
