# train_classifier.py

from __future__ import annotations

from pathlib import Path
import argparse
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_model(
    num_classes: int,
    image_size: int = 224,
    backbone_name: str = "MobileNetV2",
    fine_tune: bool = False,
) -> tf.keras.Model:
    """Tạo model CNN + Transfer Learning."""
    input_shape = (image_size, image_size, 3)

    backbone_name_lower = backbone_name.lower()

    if backbone_name_lower == "mobilenetv2":
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif backbone_name_lower == "efficientnetb0":
        backbone = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif backbone_name_lower == "resnet50":
        backbone = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError("backbone_name chỉ hỗ trợ: MobileNetV2, EfficientNetB0, ResNet50")

    backbone.trainable = bool(fine_tune)

    inputs = layers.Input(shape=input_shape)

    # Data augmentation chạy ngay trong model.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.12)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomBrightness(0.12)(x)

    x = preprocess_input(x)

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def train_classifier(
    dataset_dir: str = "data/dataset",
    output_model: str = "models/bakery_classifier.keras",
    classes_output: str = "models/classes.json",
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 30,
    backbone: str = "MobileNetV2",
    fine_tune: bool = False,
):
    """Train model nhận dạng bánh từ thư mục data/dataset/train và data/dataset/val."""
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục val: {val_dir}")

    output_model = Path(output_model)
    classes_output = Path(classes_output)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    classes_output.parent.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_ds.class_names)

    with open(classes_output, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, ensure_ascii=False, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(
        num_classes=len(class_names),
        image_size=image_size,
        backbone_name=backbone,
        fine_tune=fine_tune,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 if fine_tune else 3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(
            str(output_model),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Lưu lại lần cuối sau khi restore best weights.
    model.save(str(output_model))

    print(f"Đã lưu model: {output_model}")
    print(f"Đã lưu classes: {classes_output}")
    print("Class names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Transfer Learning nhận dạng bánh.")
    parser.add_argument("--dataset_dir", default="data/dataset")
    parser.add_argument("--output_model", default="models/bakery_classifier.keras")
    parser.add_argument("--classes_output", default="models/classes.json")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--backbone", default="MobileNetV2", choices=["MobileNetV2", "EfficientNetB0", "ResNet50"])
    parser.add_argument("--fine_tune", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classifier(
        dataset_dir=args.dataset_dir,
        output_model=args.output_model,
        classes_output=args.classes_output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        backbone=args.backbone,
        fine_tune=args.fine_tune,
    )
