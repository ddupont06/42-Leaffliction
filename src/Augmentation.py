import argparse
from typing import Dict, Union
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from logger import LOGGER


class ImageAugmenter:
    """
    Applies image augmentations to enhance dataset diversity.

    Supports rotation, scaling, blurring, contrast, illumination, and
    projective augmentations for individual images or batches in directories.
    Automates augmentation selection for class balance.

    Attributes:
        input_path (Path): Directory or file path for input images.
        max_images (int): Target image count for dataset balance (for dir).
    """

    def __init__(self, input_path: str):
        """
        Initializes with input path, setting max_images for directory inputs
        based on class balance.

        Args:
            input_path (str): Path to input image or image-containing
            directory.
        """
        self.input_path = Path(input_path)

        # Check if the input path is a directory and not empty
        if self.input_path.is_dir():
            if not any(self.input_path.iterdir()):
                LOGGER.error(f"The directory {self.input_path} is empty.")
                # If the directory is empty, max_images doesn't need to be
                # calculated
                self.max_images = 0
            else:
                # Calculate the maximum number of images for augmentation
                self.max_images = self.calculate_max_images()
        else:
            # If it's not a directory, assume it's a single image file
            # In this case, max_images doesn't need to be calculated
            self.max_images = 0

    def calculate_max_images(self) -> int:
        """
        Determines max image count for class balancing via image augmentation.

        Returns:
            int: Target images per class post-augmentation.
        """
        # Check if the input path is a directory; if not, return 0 as no
        # calculation is needed
        if not self.input_path.is_dir():
            return 0

        # Initialize a list to store the count of images in each class
        # directory
        image_counts = []

        # Iterate over each subdirectory in the input path
        for class_dir in [d for d in self.input_path.iterdir() if d.is_dir()]:
            # List image files in the class directory
            images = [f for f in class_dir.iterdir() if self.is_image_file(f)]
            # Append the count of images to the list
            image_counts.append(len(images))

        # If no images are found, return 0
        if not image_counts:
            return 0

        # Find the minimum image count among all classes
        min_images = min(image_counts)

        # The target is set to a certain factor (6 times) the minimum count
        # This factor can be adjusted based on the augmentation techniques used
        target_max_images = 6 * min_images + min_images

        return target_max_images

    def is_image_file(self, file_path: Path) -> bool:
        """
        Checks if a file is a valid image based on its extension.

        Args:
            file_path (Path): File path to check.

        Returns:
            bool: True if valid image file, False otherwise.
        """
        # Define a set of valid image file extensions
        valid_extensions = {".jpeg", ".jpg", ".png"}

        # Check if the file extension is in the set of valid image file
        # extensions
        return file_path.suffix.lower() in valid_extensions

    def rotation_image(
        self, image: np.ndarray, angle: float = 20.0
    ) -> np.ndarray:
        """
        Rotates an image around its center by a given angle.

        Args:
            image (np.ndarray): Image to rotate.
            angle (float): Rotation angle in degrees (default 20.0).

        Returns:
            np.ndarray: Rotated image.
        """
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the center of the image
        center = (width / 2, height / 2)

        # Get the rotation matrix for rotating the image around its center
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate the sine and cosine of the rotation angle (needed to adjust
        # the bounding box)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        # Compute the new bounding dimensions of the image after rotation
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # Adjust the rotation matrix to take into account the translation
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        # Perform the actual rotation and return the image
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (bound_w, bound_h),
            borderValue=(255, 255, 255),
        )

        return rotated_image

    def blur_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian blur to an image.

        Args:
            image (np.ndarray): Image to blur.

        Returns:
            np.ndarray: Blurred image.
        """
        # Define the size of the Gaussian kernel. The size should be odd.
        # A larger kernel will result in a more blurred image.
        kernel_size = (
            13,
            13,
        )  # This can be adjusted based on the desired level of blurring.

        # Apply Gaussian blur to the image
        # The third parameter (sigmaX) controls the std dev in the x direction;
        # setting it to 0 lets OpenCV automatically determine it based on the
        # kernel size.
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

        return blurred_image

    def contrast_image(
        self, image: np.ndarray, alpha: float = 1.7, beta: float = 0
    ) -> np.ndarray:
        """
        Adjusts image contrast (and optionally brightness).

        Args:
            image (np.ndarray): Image to adjust.
            alpha (float): Contrast factor (default 1.7).
            beta (float): Brightness adjustment (default 0).

        Returns:
            np.ndarray: Adjusted image.
        """
        # Apply the contrast adjustment
        # cv2.convertScaleAbs scales the image by alpha and adds beta to each
        # pixel, then converts the result to an 8-bit absolute value, which
        # ensures valid pixel intensity values.
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return adjusted_image

    def scale_image(self, image: np.ndarray, zoom: float = 1.3) -> np.ndarray:
        """
        Scales an image by a zoom factor.

        Args:
            image (np.ndarray): Image to scale.
            zoom (float): Zoom factor (default 1.3).

        Returns:
            np.ndarray: Scaled image.
        """
        # Calculate the center of the image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        # Generate the transformation matrix for scaling
        # The matrix specifies the scaling factor and the center point for the
        # scaling
        transformation_matrix = cv2.getRotationMatrix2D(center, 0, zoom)

        # Apply the scaling transformation
        # cv2.warpAffine performs the transformation based on the given matrix
        scaled_image = cv2.warpAffine(
            image, transformation_matrix, (width, height)
        )

        return scaled_image

    def illumination_image(
        self, image: np.ndarray, alpha: float = 1.0, beta: float = 80
    ) -> np.ndarray:
        """
        Adjusts the brightness of an image.

        Args:
            image (np.ndarray): Image to adjust.
            alpha (float): Brightness factor (default 1.0).
            beta (float): Brightness offset (default 80).

        Returns:
            np.ndarray: Brightness-adjusted image.
        """
        # Ensure that the adjustment does not overflow data type limits
        # Convert image to a 32-bit float, add the alpha and beta values, and
        # clip the result
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return adjusted_image

    def projective_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies projective transformation to simulate viewpoint change.

        Args:
            image (np.ndarray): Image to transform.

        Returns:
            np.ndarray: Transformed image.
        """
        # Get the dimensions of the image
        rows, cols = image.shape[:2]

        # Define corner points of the image
        src_points = np.float32(
            [[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]]
        )

        # Define destination points for the projective transformation
        # They determine the new location of the src points after transfo.
        # Adjusting these points will change the perspective effect.
        dst_points = np.float32(
            [
                [cols * 0.05, rows * 0.33],
                [cols * 0.9, rows * 0.25],
                [cols * 0.2, rows * 0.7],
                [cols * 0.8, rows * 0.9],
            ]
        )

        # Calculate the transformation matrix from the source and destination
        # points
        transformation_matrix = cv2.getPerspectiveTransform(
            src_points, dst_points
        )

        # Apply the perspective transformation to the image
        warped_image = cv2.warpPerspective(
            image,
            transformation_matrix,
            (cols, rows),
            borderValue=(255, 255, 255),
        )

        return warped_image

    def augment_dir_path(
        self, image_path: Union[Path, str], augmentations_needed: int
    ):
        """
        Applies random augms to an image, saving each with a unique filename.
        Balance the number of images in each class dir to the target count.

        Args:
            image_path (Union[Path, str]): Path to image.
            augmentations_needed (int): Number of augmentations to apply.
        """
        # Ensure image_path is a Path object
        image_path = Path(image_path)

        # Load the image from the specified path
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.error(f"Unable to read image: {image_path}")
            return

        # Define the list of available augmentation methods
        available_augmentations = [
            "Rotation",
            "Blur",
            "Contrast",
            "Scale",
            "Illumination",
            "Projective",
        ]

        # Randomly select augmentations to apply
        selected_augmentations = random.sample(
            available_augmentations, k=augmentations_needed
        )

        for augmentation in selected_augmentations:
            # Dynamically call the augmentation method
            augmented_image = getattr(self, f"{augmentation.lower()}_image")(
                image
            )

            # Generate a new filename for the augmented image
            new_file_name = (
                f"{image_path.stem}_{augmentation}{image_path.suffix}"
            )
            output_dir = image_path.parent

            # Save the augmented image to disk
            cv2.imwrite(str(output_dir / new_file_name), augmented_image)

            LOGGER.info(
                f"Applied {augmentation} augmentation "
                f"and saved as {new_file_name}."
            )

    def augment_img_path(self, image_path: Union[Path, str]):
        """
        Augments a single image with various transformations, saving each.

        Args:
            image_path (Union[Path, str]): Path to image.
        """
        # Convert the image path to a Path object if it's not already one
        image_path = Path(image_path)

        # Check if the file is an image
        if not self.is_image_file(image_path):
            LOGGER.error(
                f"The file {image_path} is not a valid image. "
                f"Supported formats are: .jpeg, .jpg, .png."
            )
            return

        # Read the image from the given path
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.error(f"Unable to read image: {image_path}")
            return

        # Dictionary to store augmented images
        augmented_images = {}

        # List of augmentation techniques to be applied
        augmentations_to_apply = [
            "Rotation",
            "Blur",
            "Contrast",
            "Scale",
            "Illumination",
            "Projective",
        ]

        # Apply each augmentation and store the result
        for augmentation in augmentations_to_apply:
            # Dynamically call the augmentation method based on the
            # augmentation name
            augmented_image = getattr(self, f"{augmentation.lower()}_image")(
                image
            )

            # Store the augmented image in the dictionary
            augmented_images[augmentation] = augmented_image

            # Generate a new filename for the augmented image
            new_file_name = (
                f"{image_path.stem}_{augmentation}{image_path.suffix}"
            )
            output_dir = image_path.parent

            # Save the augmented image to the same directory as the original
            cv2.imwrite(str(output_dir / new_file_name), augmented_image)

        # Optionally display the augmented images alongside the original
        self.display_augmented_images(image, augmented_images)

    def display_augmented_images(
        self,
        original_image: np.ndarray,
        augmented_images: Dict[str, np.ndarray],
    ):
        """
        Displays original and augmented images together in a matplotlib figure.
        Onyl works when a single image is augmented.

        Args:
            original_image (np.ndarray): The original image.
            augmented_images (Dict[str, np.ndarray]): Augmented images keyed
                                                      by technique.
        """
        # Convert the original image from BGR to RGB
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Prepare the figure for displaying images
        plt.figure(figsize=(18, 12))

        # Number of images to display (original + number of augmentations)
        num_images = 1 + len(augmented_images)

        # Display the original image
        plt.subplot(1, num_images, 1)
        plt.imshow(original_image_rgb)
        plt.title("Original")
        plt.axis("off")

        # Display each augmented image
        for i, (augmentation_name, augmented_image) in enumerate(
            augmented_images.items(), start=2
        ):
            # Convert augmented images from BGR to RGB
            augmented_image_rgb = cv2.cvtColor(
                augmented_image, cv2.COLOR_BGR2RGB
            )

            plt.subplot(1, num_images, i)
            plt.imshow(augmented_image_rgb)
            plt.title(augmentation_name)
            plt.axis("off")

        # Show the plot
        plt.show()

    def augment_images(self):
        """
        Applies augmentations to images at the specified path, based on
        whether it's a file or directory.
        """
        if self.input_path.is_file():
            # If the input path is a file, augment that single image and
            # display the results
            LOGGER.info(
                f"Starting augmentation for single image: {self.input_path}"
            )
            self.augment_img_path(self.input_path)
        elif self.input_path.is_dir():
            # If the input path is a directory, augment images within each
            # subdirectory
            LOGGER.info(
                f"Starting augmentation for directory: {self.input_path}"
            )

            # Iterate over each class directory in the input path
            for class_dir in [
                d for d in self.input_path.iterdir() if d.is_dir()
            ]:
                images = [
                    f for f in class_dir.iterdir() if self.is_image_file(f)
                ]
                current_count = len(images)
                target_count = self.max_images

                # Calculate how many additional images are needed to reach the
                # target count
                images_to_augment = max(0, target_count - current_count)

                if images_to_augment <= 0:
                    LOGGER.info(
                        f"No augmentations needed for {class_dir}. "
                        f"It already has {current_count} images."
                    )
                    continue

                LOGGER.info(
                    f"Augmenting images in {class_dir}: {current_count} "
                    f"existing, {images_to_augment} augmentations planned."
                )
                augmentations_per_image = images_to_augment // current_count
                extra_augmentations = images_to_augment % current_count

                # Apply augmentations to each image in the class directory
                for i, img_file in enumerate(images):
                    total_augmentations_for_this_image = (
                        augmentations_per_image
                        + (1 if i < extra_augmentations else 0)
                    )
                    LOGGER.info(
                        f"Performing {total_augmentations_for_this_image} "
                        f"augmentations for {img_file}"
                    )
                    self.augment_dir_path(
                        img_file, total_augmentations_for_this_image
                    )
        else:
            LOGGER.error(
                f"The path {self.input_path} is not valid. "
                f"It must be either a directory or a file."
            )


def main():
    """
    Runs the image augmentation process from command-line input.

    Usage:
        python3 Augmentation.py /path/to/image_or_directory
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Image Augmentation Tool")

    # Add an argument for the input path with a help description
    parser.add_argument(
        "path",
        type=str,
        help="Path to the image or directory to be augmented.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure the input path exists
    input_path = Path(args.path)
    if not input_path.exists():
        LOGGER.error(f"The provided path '{args.path}' does not exist.")
        return

    # Initialize the ImageAugmenter with the input path
    augmenter = ImageAugmenter(str(input_path))

    # Perform the augmentation process
    augmenter.augment_images()

    LOGGER.info("Image augmentation completed.")


if __name__ == "__main__":
    main()
