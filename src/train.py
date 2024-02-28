import os
import shutil
import zipfile
import argparse
from glob import glob

from tensorflow.keras.applications.xception import Xception

# from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_model_and_save(data_dir_path):
    # Define the diectory paths for the training and validation datasets
    train_dir = os.path.join(data_dir_path, "train")
    val_dir = os.path.join(data_dir_path, "val")
    zip_file_path = "datasets_and_model.zip"

    # Load the Xception model pre-trained on ImageNet
    base_model = Xception(include_top=False, weights="imagenet")

    # Add a global average pooling layer based on the base model's output
    avg = GlobalAveragePooling2D()(base_model.output)

    # Add a dense output layer with one unit per class and softmax
    n_classes = len(glob(os.path.join(train_dir, "*")))
    output = Dense(n_classes, activation="softmax")(avg)

    # Create the Keras model
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # # Unfreeze the first 10 layers of the base model
    # for layer in base_model.layers[:10]:
    #     layer.trainable = True

    # # Unfreeze the last 10 layers of the base model
    # fine_tune_at = len(base_model.layers) - 10
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = True

    # Set the optimizer as the SGD optimizer
    optimizer = legacy.SGD(learning_rate=0.2, momentum=0.9)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Data preparation
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load the training and validation datasets
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode="categorical",
        batch_size=32,
        keep_aspect_ratio=True,
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        class_mode="categorical",
        batch_size=32,
        keep_aspect_ratio=True,
    )

    # # lr_scheduler to reduce learning rate when val_loss no longer improves
    # lr_scheduler = ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.1, patience=2, verbose=1, min_lr=2e-4
    # )

    # Model fitting
    model.fit(
        train_generator,
        batch_size=32,
        epochs=10,
        # callbacks=[lr_scheduler],
        validation_data=val_generator,
        # steps_per_epoch=train_generator.samples // 32,
        # validation_steps=val_generator.samples // 32,
    )

    # Save the trained model
    model.save("model.keras")

    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        for root, dirs, files in os.walk(data_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Rename 'train' dir in the file path by 'augmented_images'
                zip_aug_img = file_path.replace(
                    train_dir, "augmented_directory"
                )
                # Add the file to the zip file
                zipf.write(file_path, zip_aug_img)

        zipf.write("model.keras")

    os.remove("model.keras")
    shutil.rmtree("datasets")


if __name__ == "__main__":
    """
    Trains a leaf disease classification model using Xception and saves it
    with augmented images in a zip file.

    This script accepts a directory path containing training, validation and
    test data as a command-line argument.
    The data should be organized into class-named subdirectories. It employs
    transfer learning from Xception, performs data augmentation, and packs the
    trained model and augmented images into 'datasets_and_model.zip'.

    Usage:
        python3 predict.py /path/to/directory that contains subdirectories
        with training, validation and test data
    """
    parser = argparse.ArgumentParser(
        description="Train a model to classify leaf diseases and save the "
                    "model and augmented images in a zip file"
    )
    parser.add_argument(
        "data_dir_path",
        type=str,
        help="Path to a directory containing subdirectories named by leaf "
             "disease classes.",
    )
    args = parser.parse_args()

    train_model_and_save(args.data_dir_path)
