import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 폰트 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False
DPI = 150


def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ----- 모델 구조 -----
def save_model_summary(model, txt_path):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_layer_table(model, csv_path):
    import csv

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "layer_type", "output_shape", "n_params"])
        for layer in model.layers:
            out_shape = getattr(layer, "output_shape", "N/A")
            writer.writerow([layer.name, layer.__class__.__name__, str(out_shape), layer.count_params()])


def try_plot_model_diagram(model, png_path):
    # pydot/graphviz 없으면 조용히 건너뜀 (environment.md 참고)
    try:
        import tensorflow as tf
        tf.keras.utils.plot_model(model, to_file=png_path, show_shapes=True, show_layer_names=True, dpi=150)
        return True
    except Exception as e:
        print(f"[viz] plot_model 생성 실패(pydot/graphviz 미설치 가능): {e}")
        return False


# ----- 데이터 분포 / 샘플 그리드 -----
def plot_class_distribution(counts_by_split: dict, out_path):
    classes = list(next(iter(counts_by_split.values())).keys())
    splits = list(counts_by_split.keys())
    x = np.arange(len(classes))
    width = 0.8 / len(splits)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, split in enumerate(splits):
        bars = ax.bar(x + i * width, [counts_by_split[split][c] for c in classes], width, label=split)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("샘플 수")
    ax.set_title("Split별 클래스별 샘플 수")
    ax.legend()
    _savefig(out_path)


def plot_sample_grid(images_f32, titles, out_path, suptitle, n_cols=6):
    n = len(images_f32)
    if n == 0:
        return
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(2 * n_cols, 2.2 * n_rows))
    fig.suptitle(suptitle, fontsize=14)
    for i, (img, title) in enumerate(zip(images_f32, titles)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(np.clip(img[..., :3], 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    _savefig(out_path)


def plot_augmentation_comparison(before_list, after_list, out_path, n_show=6):
    n = min(n_show, len(before_list))
    fig = plt.figure(figsize=(2 * n, 4.4))
    for i in range(n):
        ax1 = fig.add_subplot(2, n, i + 1)
        ax1.imshow(np.clip(before_list[i][..., :3], 0, 1))
        ax1.axis("off")
        ax1.set_title(f"#{i}", fontsize=8)
        ax2 = fig.add_subplot(2, n, n + i + 1)
        ax2.imshow(np.clip(after_list[i][..., :3], 0, 1))
        ax2.axis("off")
    fig.suptitle("증강 전/후 비교 (위: 전, 아래: 후)", fontsize=14)
    _savefig(out_path)


# ----- 학습 곡선 -----
def plot_training_curves(history: dict, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], label="train loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_" + acc_key
    axes[1].plot(epochs, history[acc_key], label="train acc")
    if val_acc_key in history:
        axes[1].plot(epochs, history[val_acc_key], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    if val_acc_key in history:
        gap = history[acc_key][-1] - history[val_acc_key][-1]
        fig.suptitle(f"최종 train-val accuracy 격차: {gap:+.3f} (양수가 클수록 과적합 의심)")
    _savefig(out_path)


# ----- 필터 / 특징맵 (교수님 코드 deep_cnn_filterVis.py / deep_cnn_featureMapVis.py 방식) -----
def plot_conv_filters(model, layer_name, out_path, max_filters=64):
    weights, bias = model.get_layer(layer_name).get_weights()
    f_min, f_max = weights.min(), weights.max()
    filters = (weights - f_min) / (f_max - f_min + 1e-8)

    n_filt = min(filters.shape[3], max_filters)
    n_cols = int(np.ceil(np.sqrt(n_filt)))
    n_rows = int(np.ceil(n_filt / n_cols))

    fig = plt.figure(figsize=(1.3 * n_cols, 1.3 * n_rows))
    fig.suptitle(f"'{layer_name}' 필터 시각화 (전체 {filters.shape[3]}개 중 {n_filt}개)")
    for i in range(n_filt):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        filt = filters[:, :, :, i]
        ax.imshow(filt[:, :, 0] if filt.shape[-1] == 1 else filt[:, :, :3], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    _savefig(out_path)


def plot_feature_maps(model, conv_layer_names, x_input, out_path, max_maps=32):
    import tensorflow as tf

    layer_outputs = [model.get_layer(name).output for name in conv_layer_names]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x_input, verbose=0)
    if len(conv_layer_names) == 1:
        activations = [activations]

    n_layers = len(conv_layer_names)
    fig = plt.figure(figsize=(14, 3.2 * n_layers))
    for li, (name, act) in enumerate(zip(conv_layer_names, activations)):
        n_maps = min(act.shape[-1], max_maps)
        n_cols = min(n_maps, 8)
        n_rows_this = int(np.ceil(n_maps / n_cols))
        for i in range(n_maps):
            ax = fig.add_subplot(n_layers * n_rows_this, n_cols, li * n_rows_this * n_cols + i + 1)
            ax.imshow(act[0, :, :, i], cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Conv 층별 특징맵(feature map) 시각화")
    _savefig(out_path)


# ----- 예측 분석 (교수님 코드 deep_cnn_fashion_train_confMatrix.py 방식) -----
def plot_confusion_matrix(cm, class_names, out_path, normalize=True):
    cm = np.array(cm, dtype=np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_disp, fmt = cm / row_sum, "{:.2f}"
    else:
        cm_disp, fmt = cm, "{:.0f}"

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("예측 클래스")
    ax.set_ylabel("정답 클래스")
    ax.set_title("Confusion Matrix" + (" (정규화)" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm_disp[i, j]), ha="center", va="center",
                     color="white" if cm_disp[i, j] > cm_disp.max() / 2 else "black", fontsize=9)
    fig.colorbar(im)
    _savefig(out_path)


def plot_prediction_grid(images_f32, true_names, pred_names, probs, out_path, title, n_show=16, n_cols=4):
    n = min(n_show, len(images_f32))
    if n == 0:
        return
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3 * n_cols, 3.2 * n_rows))
    fig.suptitle(title, fontsize=14)
    for i in range(n):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(np.clip(images_f32[i][..., :3], 0, 1))
        ax.axis("off")
        ax.set_title(f"정답:{true_names[i]}\n예측:{pred_names[i]} ({probs[i]:.2f})", fontsize=9)
    _savefig(out_path)
