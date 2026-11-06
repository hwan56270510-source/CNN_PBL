import os

import cv2
import numpy as np
import tensorflow as tf

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ----- 라벨/이미지 -----
def list_split_files(split_dir):
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        pairs.append((os.path.join(img_dir, fname), os.path.join(lbl_dir, stem + ".txt")))
    return pairs


def parse_label_file(label_path, img_w, img_h):
    # YOLO-seg 한 줄 = class_id x1 y1 x2 y2 ... (정규화 좌표). 빈 라벨 = 배경 이미지.
    objs = []
    if not os.path.exists(label_path):
        return objs
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(float(parts[0]))
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
            xs, ys = coords[0::2] * img_w, coords[1::2] * img_h
            polygon = np.stack([xs, ys], axis=1)
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            objs.append({"class_id": cls_id, "polygon": polygon, "bbox": bbox})
    return objs


def polygon_to_mask(polygon, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask


# ----- instance 목록 생성 (폴더=split 그대로, 재분할 없음) -----
def sample_other_bbox(img_w, img_h, size_range, rng):
    frac = rng.uniform(size_range[0], size_range[1])
    side = max(8, int(min(img_w, img_h) * frac))
    x1 = int(rng.integers(0, max(1, img_w - side)))
    y1 = int(rng.integers(0, max(1, img_h - side)))
    return np.array([x1, y1, x1 + side, y1 + side], dtype=np.float32)


def build_instance_index(split_dir, classes, other_cfg):
    pairs = list_split_files(split_dir)
    instances = []
    other_id = classes["other_class_id"]
    n_other = int(other_cfg.get("crops_per_empty_image", 2))
    size_range = other_cfg.get("crop_size_range", [0.10, 0.35])
    base_seed = int(other_cfg.get("seed", 0))

    for img_path, lbl_path in pairs:
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        objs = parse_label_file(lbl_path, img_w, img_h)

        if objs:
            for obj in objs:
                instances.append({"img_path": img_path, "class_id": obj["class_id"],
                                   "polygon": obj["polygon"], "base_bbox": obj["bbox"], "is_other": False})
        elif other_cfg.get("enable", True):
            # 빈 라벨(배경) 이미지 -> other 랜덤 crop, 파일명 기반 시드로 재현성 고정
            rng = np.random.default_rng((base_seed + abs(hash(os.path.basename(img_path)))) % (2**32))
            for _ in range(n_other):
                bbox = sample_other_bbox(img_w, img_h, size_range, rng)
                instances.append({"img_path": img_path, "class_id": other_id,
                                   "polygon": None, "base_bbox": bbox, "is_other": True})
    return instances


def class_counts(instances, classes):
    counts = {name: 0 for name in classes["id2name"].values()}
    for inst in instances:
        counts[classes["id2name"][inst["class_id"]]] += 1
    return counts


# ----- bbox / mask 가공 -----
def expand_bbox(bbox, margin, img_w, img_h):
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return np.array([max(0, x1 - dx), max(0, y1 - dy), min(img_w, x2 + dx), min(img_h, y2 + dy)], dtype=np.float32)


def jitter_bbox(bbox, center_jitter, scale_jitter_range, img_w, img_h, rng):
    # 실제 추론 땐 YOLO 예측 bbox가 들어오므로, 학습 때 bbox를 일부러 흔들어 강인하게 만듦
    x1, y1, x2, y2 = bbox
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    cx += rng.uniform(-center_jitter, center_jitter) * w
    cy += rng.uniform(-center_jitter, center_jitter) * h
    lo, hi = scale_jitter_range
    w = max(4.0, w * (1 + rng.choice([-1, 1]) * rng.uniform(lo, hi)))
    h = max(4.0, h * (1 + rng.choice([-1, 1]) * rng.uniform(lo, hi)))
    x1, x2 = np.clip([cx - w / 2, cx + w / 2], 0, img_w)
    y1, y2 = np.clip([cy - h / 2, cy + h / 2], 0, img_h)
    return np.array([x1, y1, max(x2, x1 + 2), max(y2, y1 + 2)], dtype=np.float32)


def morph_mask(mask, kernel_range, rng):
    # 마스크 경계를 침식/팽창시켜 YOLO 예측 마스크의 거친 경계를 흉내냄
    k = int(rng.integers(kernel_range[0], kernel_range[1] + 1))
    if k <= 0:
        return mask
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask, kernel) if rng.random() < 0.5 else cv2.dilate(mask, kernel)


def crop_and_letterbox(img, mask, bbox, size):
    # bbox 영역을 crop 후 비율 유지(letterbox)하며 size x size 로 맞춤
    # mask가 None이면(= other 클래스, 실제 polygon 없음) crop 전체를 1로 채운 더미 마스크 사용
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop_img = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2] if mask is not None else np.ones(crop_img.shape[:2], dtype=np.uint8)

    h, w = crop_img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    top, left = (size - new_h) // 2, (size - new_w) // 2

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(crop_img, (new_w, new_h))
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    mask_canvas[top:top + new_h, left:left + new_w] = cv2.resize(crop_mask, (new_w, new_h),
                                                                   interpolation=cv2.INTER_NEAREST)
    return canvas, mask_canvas


def build_input_array(crop_img_uint8, crop_mask_uint8, mode):
    rgb = crop_img_uint8.astype(np.float32) / 255.0
    if mode == "rgb":
        return rgb
    m = (crop_mask_uint8 > 0).astype(np.float32)[..., None]
    if mode == "masked_rgb":
        return rgb * m
    if mode == "rgbm":
        return np.concatenate([rgb, m], axis=-1)
    raise ValueError(f"알 수 없는 input_mode: {mode}")


def input_channels(mode):
    return 4 if mode == "rgbm" else 3


def augment_pixels(img_f32, mask_u8, cfg, rng):
    # crop 이후(letterbox 결과)에 적용하는 pixel-level 증강
    if cfg.get("hflip", True) and rng.random() < 0.5:
        img_f32, mask_u8 = img_f32[:, ::-1, :].copy(), mask_u8[:, ::-1].copy()

    deg = cfg.get("rotate_deg", 0)
    if deg:
        h, w = img_f32.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-deg, deg), 1.0)
        img_f32 = cv2.warpAffine(img_f32, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask_u8 = cv2.warpAffine(mask_u8, M, (w, h), flags=cv2.INTER_NEAREST)

    b, c = cfg.get("brightness_delta", 0.0), cfg.get("contrast_delta", 0.0)
    if b or c:
        img_f32 = np.clip(img_f32 * (1.0 + rng.uniform(-c, c)) + rng.uniform(-b, b), 0.0, 1.0)

    return img_f32, mask_u8


# ----- Keras Sequence -----
class ToolCropSequence(tf.keras.utils.Sequence):
    # train=True  : bbox jitter + mask 침식/팽창 + pixel 증강을 매 epoch마다 새로 적용
    # train=False : margin만 적용한 고정 crop, 증강 없음 -> val/test 재현성 보장
    def __init__(self, instances, classes, data_cfg, augment_cfg=None, batch_size=16, train=True, seed=0):
        self.instances = instances
        self.num_classes = classes["num_classes"]
        self.input_size = data_cfg["input_size"]
        self.bbox_margin = data_cfg["bbox_margin"]
        self.input_mode = data_cfg["input_mode"]
        self.augment_cfg = augment_cfg if (train and augment_cfg and augment_cfg.get("enable", False)) else None
        self.batch_size = batch_size
        self.train = train
        self.rng = np.random.default_rng(seed)
        self._img_cache = {}  # 소량 데이터 전제 -> 원본 이미지를 메모리에 캐시
        self.indices = np.arange(len(instances))
        if self.train:
            self.rng.shuffle(self.indices)

        # 클래스 불균형 대응(sample_weight, one-hot 라벨과 무관하게 항상 동작)
        counts = np.maximum(np.bincount([i["class_id"] for i in instances], minlength=self.num_classes), 1)
        self.class_weight_vec = counts.sum() / (self.num_classes * counts)

    def __len__(self):
        return max(1, int(np.ceil(len(self.instances) / self.batch_size)))

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.indices)

    def _load_image(self, path):
        if path not in self._img_cache:
            self._img_cache[path] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self._img_cache[path]

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        xs, ys, ws = [], [], []
        for i in batch_idx:
            x, y_id = self._build_one(self.instances[i])
            xs.append(x)
            ys.append(y_id)
            ws.append(self.class_weight_vec[y_id])
        X = np.stack(xs, axis=0).astype(np.float32)
        Y = tf.keras.utils.to_categorical(ys, num_classes=self.num_classes)
        return X, Y, np.array(ws, dtype=np.float32)

    def _build_one(self, inst):
        img = self._load_image(inst["img_path"])
        img_h, img_w = img.shape[:2]
        bbox = expand_bbox(inst["base_bbox"], self.bbox_margin, img_w, img_h)
        mask = polygon_to_mask(inst["polygon"], img_h, img_w) if inst["polygon"] is not None else None

        if self.augment_cfg:
            bbox = jitter_bbox(bbox, self.augment_cfg.get("bbox_jitter_center", 0.1),
                                self.augment_cfg.get("bbox_jitter_scale", [0.1, 0.2]), img_w, img_h, self.rng)
            if mask is not None and self.augment_cfg.get("mask_morph", False):
                mask = morph_mask(mask, self.augment_cfg.get("mask_morph_kernel_range", [1, 3]), self.rng)

        crop_img, crop_mask = crop_and_letterbox(img, mask, bbox, self.input_size)
        if crop_img is None:  # bbox가 이미지 밖으로 완전히 밀린 극단적 경우
            crop_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            crop_mask = np.zeros((self.input_size, self.input_size), dtype=np.uint8)

        img_f32 = crop_img.astype(np.float32) / 255.0
        if self.augment_cfg:
            img_f32, crop_mask = augment_pixels(img_f32, crop_mask, self.augment_cfg, self.rng)

        x = build_input_array(np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8), crop_mask, self.input_mode)
        return x, inst["class_id"]


def build_dataset_arrays(instances, classes, data_cfg):
    # eval/시각화용: 증강 없이 전체 instance를 한 번에 메모리로 로드 (소량 데이터 전제)
    seq = ToolCropSequence(instances, classes, data_cfg, augment_cfg=None, batch_size=1, train=False)
    xs, ys, metas = [], [], []
    for inst in instances:
        x, y_id = seq._build_one(inst)
        xs.append(x)
        ys.append(y_id)
        metas.append({"img_path": inst["img_path"], "class_id": y_id, "is_other": inst["is_other"]})
    n_ch = input_channels(data_cfg["input_mode"])
    X = np.stack(xs, axis=0).astype(np.float32) if xs else np.zeros((0, data_cfg["input_size"],
                                                                       data_cfg["input_size"], n_ch))
    Y = tf.keras.utils.to_categorical(ys, num_classes=classes["num_classes"]) if ys else np.zeros(
        (0, classes["num_classes"]))
    return X, Y, metas
