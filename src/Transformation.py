import random
from typing import Dict, Union
from pathlib import Path

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from logger import LOGGER
from plantcv import plantcv as pcv


class Options:
    """
    Stores options for image processing.

    Attributes:
        path (Path): Source directory or image file.
        dst (Path): Destination directory for processed images.
        trs (str): Transformation type, default is "all".
        is_single_image (bool): True if source is a single image, False if not.
    """

    def __init__(self, src: str, dst: str = None, trs: str = "all"):
        """
        Initializes the Options instance with source, destination,
        transformation type, and is_single_image flag.
        """
        self.path = Path(src)
        self.dst = Path(dst) if dst else self.path.parent
        self.trs = trs
        self.is_single_image = self.path.is_file()


class ImageTransformer:
    """
    Processes images using PlantCV, supporting individual or batch processing.

    Attributes:
        opt (Options): Configuration for processing including path,
                       destination, and transformations.
    """

    def __init__(self, options: Options):
        """
        Initializes with options, validates the src path, and processes images.
        """
        self.opt = options
        # Validate source path
        if not self.opt.path.exists():
            raise ValueError(f"Img or dir path doesn't exist: {self.opt.path}")
        if self.opt.path.is_dir() and not list(self.opt.path.glob("*")):
            raise ValueError(f"Directory is empty: {self.opt.path}")

        # Validate -dst and -trs options for single image
        if self.opt.path.is_dir() and (
            self.opt.dst != self.opt.path.parent or self.opt.trs != "all"
        ):
            raise ValueError(
                "-dst and -trs options can only be used when "
                "processing a single image."
            )

        if self.opt.is_single_image and self.is_supported_image(self.opt.path):
            # Process single image
            self.transform_img_path(self.opt.path)
        elif self.opt.path.is_dir():
            # Process directory of images
            self.max_images = self.calculate_max_images()
            self.transform_dir_path()
        else:
            raise ValueError(
                "Invalid path type or unsupported file format. Must be either "
                "a directory or a supported image file (.jpeg, .jpg, .png.)"
            )

    def calculate_max_images(self) -> int:
        """
        Calculates the target image count for dataset balancing.

        Returns:
            int: Target image count.
        """
        # Initialize a list to store the count of images in each class dir
        image_counts = []

        # Iterate over each subdirectory (class) in the dataset
        for class_dir in [d for d in self.opt.path.iterdir() if d.is_dir()]:
            # List all supported image files in the class directory
            images = [f for f in class_dir.iterdir() if
                      self.is_supported_image(f)]
            # Append the count of images to the list
            image_counts.append(len(images))

        # If no images are found in any class, return 0
        if not image_counts:
            return 0

        # Find the minimum image count among all classes
        min_images = min(image_counts)

        # The target is set to a certain factor (6 times) the minimum count
        # This factor can be adjusted based on the transfo techniques used
        target_max_images = 6 * min_images + min_images

        return target_max_images

    @staticmethod
    def is_supported_image(file_path: Path) -> bool:
        """
        Checks if the file is in a supported image format.

        Args:
            file_path (Path): File path to check.

        Returns:
            bool: True if supported, False otherwise.
        """
        # Define a set of valid image file extensions
        valid_extensions = {".jpg", ".jpeg", ".png"}

        # Check if the file extension is in the set of valid image file
        return file_path.suffix.lower() in valid_extensions

    def canny_edge(self, img):
        """
        Applies Canny edge detection.
        """
        edge = pcv.canny_edge_detect(img=img, high_thresh=145)
        return edge

    def gaussian_blur(self, img):
        """
        Applies Gaussian blur.
        """
        # Convert to grayscale colorspace
        l_channel = pcv.rgb2gray_lab(rgb_img=img, channel="l")
        # Threshold/segment plant from background
        l_thresh = pcv.threshold.binary(
            gray_img=l_channel, threshold=120, object_type="dark"
        )
        # Fill small objects (speckles)
        l_clean = pcv.fill(bin_img=l_thresh, size=15)
        return l_clean

    def mask(self, img):
        """
        Applies a binary mask to an image.
        """
        binary_mask = self.gaussian_blur(img)
        # Apply binary 'white' gaussian blur mask to image
        mask = pcv.apply_mask(img=img, mask=binary_mask, mask_color="white")
        return mask

    def roi(self, img):
        """
        Defines an ROI (region of interest) for the image.
        """
        masked_img = self.mask(img)
        # ROI is defined as a rectangle
        roi = pcv.roi.rectangle(
            img=masked_img,
            x=0,
            y=0,
            h=256,
            w=256,
            # img=masked_img, x=0, y=0, h=300, w=301 # For test_subject_image
        )
        return roi

    def labeled_mask(self, img):
        """
        Make a new filtered mask that only keeps the plant in your ROI and not
        objects outside of the ROI.
        """
        binary_mask = self.gaussian_blur(img)
        roi = self.roi(img)
        labeled_mask = pcv.roi.filter(mask=binary_mask, roi=roi,
                                      roi_type="cutto")
        return labeled_mask

    def analyze_object(self, img):
        """
        Find shape properties.
        """
        labeled_mask = self.labeled_mask(img)
        shape = pcv.analyze.size(img=img, labeled_mask=labeled_mask,
                                 n_labels=1)
        return shape

    def roi_objects(self, img):
        """
        Defines ROI for objects.
        """
        labeled_mask = self.labeled_mask(img)
        boundary = pcv.analyze.bound_horizontal(
            img=img, labeled_mask=labeled_mask, line_position=0
        )
        return boundary

    def pseudolandmarks(self, img):
        """
        Define pseudolandmarks for the image.
        """
        labeled_mask = self.labeled_mask(img)

        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=img, mask=labeled_mask, label=None
        )

        annotated_img = img.copy()
        for i in top:
            cv2.circle(annotated_img, (int(i[0][0]), int(i[0][1])), 5,
                       (0, 0, 255), -1)
        for i in bottom:
            cv2.circle(annotated_img, (int(i[0][0]), int(i[0][1])), 5,
                       (0, 255, 0), -1)
        for i in center_v:
            cv2.circle(annotated_img, (int(i[0][0]), int(i[0][1])), 5,
                       (255, 0, 0), -1)

        return annotated_img

    def apply_transformations(self, img):
        """
        Apply all transformations to the image.
        Used when the user specifies a single image.
        """
        transformations_mapping = {
            "Original": lambda img: img,
            "Canny_Edge": self.canny_edge,
            "Gaussian_Blur": self.gaussian_blur,
            "Mask": self.mask,
            "Roi_Objects": self.roi_objects,
            "Analyze_Object": self.analyze_object,
            "Pseudolandmarks": self.pseudolandmarks,
        }
        # Apply all transformations if 'all' is specified
        if self.opt.trs.lower() == "all":
            transformations = {
                name: method(img) for name, method in
                transformations_mapping.items()
            }
        else:
            # Find the correct transfo method in a case-insensitive manner
            # while preserving the original case in the keys
            # e.g., 'canny_edge' -> 'Canny_Edge', etc.
            transfo_key = next(
                (
                    key
                    for key in transformations_mapping
                    if key.lower() == self.opt.trs.lower()
                ),
                None,
            )
            # Apply the specified transformation
            if transfo_key:
                # Get the transformation method from the mapping
                transformation_method = transformations_mapping[transfo_key]
                # Apply the transformation to the image
                transformations = {transfo_key:
                                   transformation_method(img)}
            else:
                raise ValueError(
                    f"Specified transformation '{self.opt.trs}' is not "
                    f"supported."
                )

        return transformations

    def save_transformations(self, transformations, base_filename):
        """
        Save the transformations of a single image.
        Used when the user specifies a single image.
        """
        self.opt.dst.mkdir(parents=True, exist_ok=True)
        for key, img in transformations.items():
            filename = f"{base_filename}_{key}.jpg"
            save_path = self.opt.dst / filename
            pcv.print_image(img, str(save_path))
            LOGGER.info(f"Saved: {save_path}")

    def display_img_path(self, transformations: Dict[str, np.ndarray]):
        """
        Display the transformations of a single image.
        Used when the user specifies a single image.
        """
        plt.figure(figsize=(15, 10))
        fig_titles = {
            "Original": "Figure IV.1: Original",
            "Canny_Edge": "Figure IV.2: Canny Edge",
            "Gaussian_Blur": "Figure IV.3: Gaussian Blur",
            "Mask": "Figure IV.4: Mask",
            "Roi_Objects": "Figure IV.5: ROI Objects",
            "Analyze_Object": "Figure IV.6: Analyze Object",
            "Pseudolandmarks": "Figure IV.7: Pseudolandmarks",
        }
        for i, (title, transformed_img) in enumerate(transformations.items(),
                                                     start=1):
            plt.subplot(2, 4, i)
            plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
            plt.title(fig_titles[title], y=-0.1)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def transform_img_path(self, image_path):
        """
        Process a single image and display the transformations.
        Used when the user specifies a single image.
        """
        img, _, _ = pcv.readimage(str(image_path))
        if img is None:
            LOGGER.error(f"Unable to read image: {image_path}")
            return

        # # For debugging purposes
        # pcv.params.debug = "print"
        # colorspaces = pcv.visualize.colorspaces(rgb_img=img,
        #                                         original_img=False)
        # l_channel = pcv.rgb2gray_lab(rgb_img=img, channel="l")
        # hist = pcv.visualize.histogram(img=l_channel, bins=25)
        # l_thresh = pcv.threshold.binary(
        #     gray_img=l_channel, threshold=120, object_type="dark"
        # )
        # l_clean = pcv.fill(bin_img=l_thresh, size=15)

        transformations = self.apply_transformations(img)
        self.save_transformations(transformations, image_path.stem)
        self.display_img_path(transformations)

    def transform_imgs_in_dir(
        self, image_path: Union[Path, str], transformations_needed: int
    ):
        """
        Apply a random set of transformations to an image.
        Used when the user specifies a directory.
        """
        image_path = Path(image_path)
        img, _, _ = pcv.readimage(str(image_path))
        if img is None:
            LOGGER.error(f"Unable to read image: {image_path}")
            return

        transformations = [
            "Canny_Edge",
            "Gaussian_Blur",
            "Mask",
            "Roi_Objects",
            "Analyze_Object",
            "Pseudolandmarks",
        ]
        random.shuffle(transformations)

        for i in range(transformations_needed):
            suffix = transformations[i % len(transformations)]
            transformed_image = getattr(self, f"{suffix.lower()}")(img)
            new_file_name = f"{image_path.stem}_{suffix}{image_path.suffix}"
            output_dir = image_path.parent
            pcv.print_image(transformed_image, str(output_dir / new_file_name))

    def transform_dir_path(self):
        """
        Apply transformations to all images in the directory.
        Transformations are applied to balance the dataset.
        Used when the user specifies a directory.
        """
        if self.opt.path.is_dir():
            # Iterate over each subdirectory (class) in the dataset
            for class_dir in [d for d in self.opt.path.iterdir() if
                              d.is_dir()]:
                # List all supported image files in the class directory
                images = [f for f in class_dir.iterdir() if
                          self.is_supported_image(f)]
                # Calculate the number of transfo needed to balance the dataset
                current_count = len(images)
                target_count = self.max_images

                images_to_transform = target_count - current_count

                if images_to_transform <= 0:
                    continue

                transfor_per_image = images_to_transform // current_count
                extra_transformations = images_to_transform % current_count

                # Apply transformations to each image in the class directory
                for i, img_file in enumerate(images):
                    total_transformations_for_this_image = transfor_per_image
                    + (1 if i < extra_transformations else 0)
                    LOGGER.info(
                        f"Performing {total_transformations_for_this_image} "
                        f"transformations for {img_file}"
                    )
                    self.transform_imgs_in_dir(
                        img_file, total_transformations_for_this_image
                    )
        else:
            LOGGER.error(
                f"The path {self.input_path} is not a directory or "
                f"a supported image file."
            )


def main():
    """
    Runs the image transformation process from command-line input.
    The transformations are based on the PlantCV library.

    Usage:
        python Transformation.py -src /path/to/image_or_directory
        -dst /path/to/save/transformed_images -trs transformation_type
    """
    parser = argparse.ArgumentParser(
        description="Transform images using PlantCV library"
    )
    parser.add_argument(
        "-src",
        "--source",
        type=str,
        required=True,
        help="The path to the image or directory containing "
             "the images to transform",
    )
    parser.add_argument(
        "-dst",
        "--destination",
        type=str,
        help="The path to the directory to save the transformed images",
    )
    parser.add_argument(
        "-trs",
        "--transformation",
        type=str,
        default="all",
        help="Specify an image transformation type: gaussian_blur, mask, "
             "analyze_object, roi_objects or pseudolandmarks",
    )
    args = parser.parse_args()

    opt = Options(src=args.source, dst=args.destination,
                  trs=args.transformation)
    try:
        ImageTransformer(opt)
    except ValueError as e:
        LOGGER.error(e)


if __name__ == "__main__":
    main()
