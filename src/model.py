import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(input_shape, num_classes, cfg):
    # 메인 모델: 소량 데이터용으로 필터 수를 작게 유지(16->32->64), GAP으로 FC 파라미터 절약
    filters_list = cfg.get("conv_blocks", [16, 32, 64])
    k = cfg.get("kernel_size", 3)
    use_bn = cfg.get("use_bn", True)
    dropout = cfg.get("dropout", 0.3)
    fc_units = cfg.get("fc_units", 64)
    head = cfg.get("head", "gap")

    model = models.Sequential(name="tool_custom_cnn")
    model.add(layers.Input(shape=input_shape))

    for i, filters in enumerate(filters_list):
        model.add(layers.Conv2D(filters, (k, k), padding="same", name=f"conv{i+1}"))  # 컨볼루션 층
        if use_bn:
            model.add(layers.BatchNormalization(name=f"bn{i+1}"))
        model.add(layers.Activation("relu", name=f"relu{i+1}"))
        model.add(layers.MaxPooling2D((2, 2), name=f"pool{i+1}"))  # 풀링 층

    if head == "flatten":
        model.add(layers.Flatten(name="flatten"))
    else:
        model.add(layers.GlobalAveragePooling2D(name="gap"))  # Flatten 대신 GAP으로 FC 입력 차원 축소

    model.add(layers.Dropout(dropout, name="dropout_fc"))
    model.add(layers.Dense(fc_units, activation="relu", name="fc1"))  # 완전연결층
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))  # 출력층
    return model


def build_mobilenet_v2(input_shape, num_classes, cfg):
    # 비교용 전이학습 베이스라인. 3채널(rgb/masked_rgb) 입력만 지원 (ImageNet 사전학습 전제)
    if input_shape[-1] != 3:
        raise ValueError("mobilenet_v2는 3채널(rgb 또는 masked_rgb) 입력만 지원합니다.")

    alpha = cfg.get("alpha", 0.35)
    freeze_base = cfg.get("freeze_base", True)
    fc_units = cfg.get("fc_units", 64)

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)  # [0,1] -> [-1,1]

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, alpha=alpha, include_top=False, weights="imagenet"
    )
    base.trainable = not freeze_base
    x = base(x, training=not freeze_base)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout_fc")(x)
    x = layers.Dense(fc_units, activation="relu", name="fc1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="tool_mobilenet_v2_baseline")


def build_model(model_cfg, input_shape, num_classes):
    mtype = model_cfg.get("type", "custom_cnn")
    if mtype == "custom_cnn":
        return build_custom_cnn(input_shape, num_classes, model_cfg.get("custom_cnn", {}))
    if mtype == "mobilenet_v2":
        return build_mobilenet_v2(input_shape, num_classes, model_cfg.get("mobilenet_v2", {}))
    raise ValueError(f"알 수 없는 model.type: {mtype}")


def compile_model(model, train_cfg):
    lr = train_cfg.get("learning_rate", 0.001)
    if train_cfg.get("optimizer", "adam") == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        loss=train_cfg.get("loss", "categorical_crossentropy"),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def get_conv_layer_names(model):
    return [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)]
