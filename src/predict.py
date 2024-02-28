import argparse
import os
import random
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from logger import LOGGER

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]


def unzip_datasets_and_model():
    """
    Checks if the 'datasets' directory and 'model.keras' file exist. If not,
    it unzips 'datasets_and_model.zip' to extract these required resources.
    """
    required_paths = ["datasets", "model.keras"]
    zip_file = "datasets_and_model.zip"
    need_extraction = any(not os.path.exists(path) for path in required_paths)
    if need_extraction:
        print("Extracting datasets and model...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall()
        print("Extraction completed.")


def validate_path(image_path):
    """
    Validates if the given image_path exists and is in a supported format.

    Parameters:
    - image_path (Path): The path to the image to be validated.

    Raises:
    - ValueError: If the file does not exist or is not in a supported format.
    """
    if not image_path.is_file():
        raise ValueError(f"The file {image_path} does not exist")
    elif image_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"The file {image_path} is not a supported image file "
            f"(.jpg, .jpeg, .png)"
        )


def load_model():
    """
    Loads the neural network model from a .keras file.

    Returns:
    - The loaded TensorFlow Keras model.
    """
    import tensorflow as tf

    model_file = "model.keras"
    return tf.keras.models.load_model(model_file)


def prepare_leaf_image(image_path):
    """
    Prepares a leaf image for prediction by resizing and normalizing it.

    Parameters:
    - image_path (str): The path to the leaf image.

    Returns:
    - A numpy array of the processed image ready for model prediction.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    loaded_img = load_img(image_path, target_size=(224, 224))
    img_array_normalized = img_to_array(loaded_img) / 255.0
    return np.expand_dims(img_array_normalized, axis=0)


def predict(image_path, model):
    """
    Predicts the class of a leaf image using the provided model.

    Parameters:
    - image_path (Path): The path to the leaf image.
    - model: The TensorFlow Keras model used for prediction.

    Returns:
    - The index of the predicted class.
    """
    prepared_image = prepare_leaf_image(image_path)
    prediction_results = model.predict(prepared_image, verbose=1)
    return np.argmax(prediction_results, axis=1)[0]


def select_random_augmentation(image_path):
    """
    Selects a random augmentation of the given image if available.

    Parameters:
    - image_path (str): The path to the original image.

    Returns:
    - The path to the augmented image or None if no augmentation is found.
    """
    augmentations = [
        "_Blur",
        "_Contrast",
        "_Illumination",
        "_Projective",
        "_Rotation",
        "_Scale",
    ]
    base_path, extension = os.path.splitext(image_path)
    for _ in range(10):
        aug_choice = random.choice(augmentations)
        augmented_path = f"{base_path}{aug_choice}{extension}"
        if Path(augmented_path).is_file():
            return augmented_path
    return None


def show_images(original_image, augmented_image=None, prediction_text=""):
    """
    Displays the original and optionally an augmented img with prediction text.

    Parameters:
    - original_image (str): Path to the original image.
    - augmented_image (str, optional): Path to the augmented image.
    - prediction_text (str, optional): Text to display as prediction result.
    """
    if augmented_image:
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        image_paths = [original_image, augmented_image]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 5))
        ax = [ax]  # Make ax a list to keep consistency
        image_paths = [original_image]

    for idx, image_path in enumerate(image_paths):
        img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ax[idx].imshow(img_rgb)
        ax[idx].axis("off")

    ax[0].set_title("Original Image", fontsize=16, weight="bold")
    if augmented_image:
        augmentation_method = augmented_image.split("_")[-1].split(".")[0]
        ax[1].set_title(
            f"Augmented: {augmentation_method}", fontsize=16, weight="bold"
        )

    display_text = f"Class predicted: {prediction_text}"
    plt.figtext(
        0.5,
        0.02,
        display_text,
        fontsize=15,
        color="red",
        ha="center",
        weight="bold",
    )
    plt.tight_layout()
    plt.show()


def evaluate_model_on_test_dataset(model):
    """
    Evaluates the model on the test dataset and prints the accuracy and loss.

    Parameters:
    - model: The TensorFlow Keras model to be evaluated.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    data_dir_path = "datasets"
    test_dir = os.path.join(data_dir_path, "test")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")


def main():
    """
    Main function to parse command line arguments and perform image
    classification or model evaluation based on the input. If an image path is
    provided, it predicts the disease class of the leaf image. If no image
    path is given, it evaluates the model on a test dataset located in the
    'datasets' directory.

    Usage:
        python3 predict.py /path/to/image or python3 predict.py (to evaluate
        the model on a 'test' folder in 'datasets' directory)
    """
    parser = argparse.ArgumentParser(
        description="Predict the disease class of a selected leaf image or "
                    "evaluate the model on a test dataset."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        nargs="?",
        help="The path to the leaf image to be classified. Leave empty to "
             "evaluate the model on a test dataset.",
    )
    args = parser.parse_args()

    if args.image_path:
        try:
            validate_path(args.image_path)
            model = load_model()
            augmented_image_path = select_random_augmentation(
                str(args.image_path)
            )
            predicted_category_index = predict(args.image_path, model)
            leaf_disease_categories = [
                "Apple_Black_rot",
                "Apple_healthy",
                "Apple_rust",
                "Apple_scab",
                "Grape_Black_rot",
                "Grape_Esca",
                "Grape_healthy",
                "Grape_spot",
            ]
            prediction_text = leaf_disease_categories[predicted_category_index]
            show_images(
                str(args.image_path), augmented_image_path, prediction_text
            )
        except Exception as e:
            LOGGER.error(e)
    else:
        zip_file_path = "datasets_and_model.zip"
        if not os.path.exists(zip_file_path):
            LOGGER.error(
                f"Error: The file '{zip_file_path}' is not present in the "
                "directory. Please ensure that the file exists before "
                "executing the command.")
            return
        else:
            unzip_datasets_and_model()
            model = load_model()
            evaluate_model_on_test_dataset(model)


if __name__ == "__main__":
    main()
