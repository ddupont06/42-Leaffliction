import os
import shutil
import random


def create_dataset(main_folder, test_size, val_size):
    """
    Splits images from a main folder into train, test, and validation datasets
    without altering the main folder. Ratio of 80% train, 10% test, and 10%
    validation by default.

    Parameters:
    - main_folder: Path to main folder containing class subfolders with images.
    - test_size: Number of images to include in the test set for each class.
    - val_size: Number of images to include in the val set for each class.
    """
    # Path to the datasets directory at the script's launch root
    dataset_root = os.path.join("datasets")
    os.makedirs(dataset_root, exist_ok=True)

    # List directories in the main folder, each considered a class
    classes = [
        d
        for d in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, d))
    ]
    for cls in classes:
        class_path = os.path.join(main_folder, cls)
        images = os.listdir(class_path)

        # Shuffle images to randomize selection for splitting
        random.shuffle(images)

        # Split images into test, validation, and train sets
        test_images = images[:test_size]
        val_images = images[test_size: test_size + val_size]
        train_images = images[test_size + val_size:]

        # Create directories for train, test, and val sets for each class
        for set_type in ["train", "test", "val"]:
            set_path = os.path.join(dataset_root, set_type, cls)
            os.makedirs(set_path, exist_ok=True)

        # Copy images to their respective directories instead of moving them
        for img in test_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(dataset_root, "test", cls, img),
            )
        for img in val_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(dataset_root, "val", cls, img),
            )
        for img in train_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(dataset_root, "train", cls, img),
            )


if __name__ == "__main__":
    # Path to the main folder containing class subfolders with images
    main_folder = "./leaves/images"
    # Ratio of about 10% on the directory with the lowest nb of images (275)
    test_size = 23
    # Ratio of about 10% on the directory with the lowest nb of images (275)
    val_size = 23

    create_dataset(main_folder, test_size, val_size)
    print("Dataset creation complete.")
