"""
train_model.py
----------------
Trains a malaria blood-smear classifier using transfer learning on
ResNet50 (pre-trained on ImageNet), as described in the project report:
- Freeze the first 40 layers, fine-tune the rest
- Custom head: GlobalAveragePooling -> Dense(256, relu) -> Dropout(0.5) -> Dense(1, sigmoid)

DATASET
-------
This script expects an image dataset arranged like this (Keras'
"flow_from_directory" layout):

    dataset/
        Parasitized/   <- positive samples (blood smears with parasites)
        Uninfected/     <- negative samples (parasite-free)

A ready-made public dataset that matches this exact layout is the
NIH "Malaria Cell Images" dataset (27,558 images), free to download from
Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Download it, unzip it, and point --data_dir at the folder that directly
contains "Parasitized" and "Uninfected".

USAGE
-----
    python train_model.py --data_dir dataset/ --epochs 10

The trained model is saved to model/malaria_resnet50.h5
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model():
    """Build ResNet50 + custom classification head, matching the report."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze the first 40 layers, fine-tune the rest (report section 3.4)
    for layer in base_model.layers[:40]:
        layer.trainable = False
    for layer in base_model.layers[40:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_generators(data_dir):
    """Create train/validation generators with a 70/15 split and augmentation."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.15,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
    )

    return train_gen, val_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to folder containing Parasitized/ and Uninfected/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", default="model/malaria_resnet50.h5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    train_gen, val_gen = get_generators(args.data_dir)
    print("Class indices (0/1 mapping):", train_gen.class_indices)

    if os.path.exists(args.output):
        print(f"Found existing checkpoint at {args.output} -- resuming from it instead of starting over.")
        model = tf.keras.models.load_model(args.output)
    else:
        model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()