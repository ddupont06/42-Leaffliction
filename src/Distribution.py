import argparse
import os
from typing import Dict, List
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from logger import LOGGER


def collect_image_paths_by_subdir(main_dir: str) -> Dict[str, List[str]]:
    """
    Collects paths of images contained within each subdirectory of a specified
    main directory.

    Args:
        main_dir (str): The path to the main directory containing
                        subdirectories of images.

    Returns:
        Dict[str, List[str]]: A dictionary where each key is the name of a
                              subdirectory and its value is a list of image
                              paths within that subdirectory.
    """
    # List all subdirectories in the main directory
    sub_dirs = os.listdir(main_dir)

    # Collect image paths for each subdirectory
    image_paths_by_subdir = {
        sub_dir: [
            os.path.join(main_dir, sub_dir, img_name)
            for img_name in os.listdir(os.path.join(main_dir, sub_dir))
        ]
        for sub_dir in sub_dirs
    }

    return image_paths_by_subdir


def count_images_by_subdir(
    image_paths_by_subdir: Dict[str, List[str]]
) -> Dict[str, int]:
    """
    Counts the number of images in each subdirectory.

    Args:
        image_paths_by_subdir (Dict[str, List[str]]): A dictionary with
                                                      subdirectory names as
                                                      keys and lists of image
                                                      paths as values.

    Returns:
        Dict[str, int]: A dictionary where each key is a subdirectory name and
                        its value is the count of images within that subdir.
    """
    # Count images for each subdirectory
    return {
        sub_dir: len(images)
        for sub_dir, images in image_paths_by_subdir.items()
    }


def plot_distribution_charts(
    image_counts_per_class: Dict[str, int],
    sample_image_paths_per_class: Dict[str, List[str]],
    num_samples: int = 1,
) -> None:
    """
    Plots distribution charts showing the class distribution and number of
    images per class, with dynamic adjustments to ensure sample images are not
    cut off and y-axis ticks are displayed as integers.

    Args:
        image_counts_per_class (Dict[str, int]): A dictionary with class names
                                                 as keys and the number of
                                                 images as values.
        sample_image_paths_per_class (Dict[str, List[str]]): A dictionary with
                                                             class names as
                                                             keys and lists of
                                                             image paths as
                                                             values.
        num_samples (int, optional): The number of sample images to include on
                                     the bar chart. Defaults to 1.
    """
    colors = plt.cm.tab20.colors[: len(image_counts_per_class)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Plot a pie chart for class distribution
    ax1.pie(
        image_counts_per_class.values(),
        labels=image_counts_per_class.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    ax1.set_title("Class Distribution")

    # Plot a bar chart for the number of images per class
    bars = ax2.bar(
        image_counts_per_class.keys(),
        image_counts_per_class.values(),
        color=colors,
    )
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Number of Images")
    ax2.set_title("Number of Images per Class")
    ax2.tick_params(axis="x", rotation=45)

    # Format y-axis ticks as integers
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # Add sample images to the bar chart and calc the necessary extra space
    tallest_bar_height = 0
    for bar, category in zip(bars, image_counts_per_class.keys()):
        add_sample_image_to_bar(bar, sample_image_paths_per_class[category][0])
        # Update tallest_bar_height if this bar is taller
        tallest_bar_height = max(tallest_bar_height, bar.get_height())

    # Dynamically calculate space needed based on the height of the tallest bar
    thumbnail_height = 33  # Assuming thumbnail height is 33 pixels
    pixels_per_unit = (
        fig.dpi
        * ax2.get_window_extent().height
        / fig.get_window_extent().height
        / ax2.viewLim.height
    )
    extra_space_units = (
        thumbnail_height / pixels_per_unit / 2
    )  # Convert pixels to plot units, adjust divisor as needed

    # Adjust the y-axis limit to accommodate the sample img on the tallest bar
    ax2.set_ylim(0, tallest_bar_height + extra_space_units)

    plt.tight_layout()
    plt.show()


def add_sample_image_to_bar(bar: plt.Rectangle, image_path: str) -> None:
    """
    Adds a sample image to a bar in a bar chart by overlaying an image on top
    of the bar.
    The image is resized to fit within a specified size and placed at the top
    center of the bar.

    Args:
        bar (plt.Rectangle): The bar to which the sample image will be added.
                             This object provides the position and size of the
                             bar within the plot.
        image_path (str): The path to the sample image file to be displayed on
                          the bar.
    """
    # Open and resize the image to fit on the bar
    img = Image.open(image_path).convert("RGBA")
    img.thumbnail((50, 50), Image.LANCZOS)  # Resize image to fit

    # Create an OffsetImage object for embedding within the plot
    img_box = OffsetImage(img, zoom=1)

    # Calculate the image's position: centered above the bar
    # The y-coordinate is set above the bar by a fixed offset
    xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
    xybox = (0.0, 50.0)  # Fixed offset in points above the bar

    # Create an AnnotationBbox for the image and add it to the bar's axes
    ab = AnnotationBbox(
        img_box,
        xy,
        xybox=xybox,
        xycoords="data",
        boxcoords="offset points",
        pad=0.5,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    bar.axes.add_artist(ab)


def main():
    """
    Visualizes distribution of images across classes in a dataset.
    Parses a directory path, calculates image counts per class, and plots
    distribution charts.

    Usage:
        python3 Distribution.py /path/to/directory
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Visualize image class distribution."
    )

    # Add an argument for the directory path with a help description
    parser.add_argument(
        "directory_path", type=str, help="Path to the image directory."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure the directory path exists and is a directory
    directory_path = Path(args.directory_path)
    if not directory_path.exists() or not directory_path.is_dir():
        LOGGER.error(
            f"The provided path '{args.directory_path}' is not a valid dir."
        )
        return

    # Collect image paths by subdirectory
    image_paths_by_subdir = collect_image_paths_by_subdir(directory_path)
    image_counts_per_class = count_images_by_subdir(image_paths_by_subdir)

    # Check for minimum required images for visualization
    if sum(image_counts_per_class.values()) < 6:
        LOGGER.error(
            "Not enough images to visualize distribution (min 6 required)."
        )
        return

    # Plot distribution charts
    plot_distribution_charts(image_counts_per_class, image_paths_by_subdir)

    LOGGER.info("Image distribution visualization completed.")


if __name__ == "__main__":
    main()
