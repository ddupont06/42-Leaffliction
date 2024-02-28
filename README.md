# Leaf Disease Detection and Analysis Toolkit

This comprehensive suite of Python scripts is designed for the processing, analysis, and prediction of leaf diseases from images. Utilizing advanced image processing and machine learning techniques, it offers valuable insights and aids in the accurate identification of various plant diseases.

## Usage

Comprising several scripts, each tailored for specific tasks such as data distribution analysis, image augmentation, image transformation, training model, and making predictions. Below is an overview of the primary functionalities and how to use them:

- **Distribution Analysis (`Distribution.py`):** Analyzes and visualizes the distribution of leaf images across different disease categories.

    ```python
    python3 Distribution.py <dir_path>
    ```
    - `<dir_path>`: Directory containing subdirectories with images of specific classes.

---

- **Data Preparation (`data_split.py`):** Prepares and splits a dataset into training, testing, and validation sets. See the data_split.py file for more details.

    ```python
    python3 data_split.py
    ```

---

- **Image Augmentation (`Augmentation.py`):** Applies image augmentation techniques to increase the diversity of the training dataset. This script supports operations on both individual images and directories of images.

    ```python
    python3 Augmentation.py <image_or_directory_path>
    ```
    - `<image_or_directory_path>`: Path to a single image or a directory containing images for augmentation.

---

- **Image Transformation (`Transformation.py`):** Transforms images using specified methods. This script supports operations on both individual images and directories of images but -dst and -trs only apply for single image path.

    ```python
    python3 Transformation.py -src <image_or_directory_path> -dst <destination_path> -trs <transformation_method>
    ```
    - `<image_or_directory_path>`: Path to the image or directory for transformation.
    - `<destination_path>`: Path where the transformed images will be saved.
    - `<transformation_method>`: Specify the transformation method to use (e.g., `gaussian_blur`, `mask`, `analyze_object`, `roi_objects`, or `pseudolandmarks`).

---

- **Model Training (`train.py`):** Trains the machine learning model on a specified dataset.

    ```python
    python3 train.py <dataset_dir>
    ```
    - `<dataset_dir>`: Directory containing the training, validation, and test data folders.

---

- **Disease Prediction (`predict.py`):** Predicts the disease category for a given leaf image or evaluates the
    model on a test dataset.

    ```python
    python3 predict.py <image_path> or python3 predict.py (to evaluate the model on a 'test' folder in 'datasets' directory))
    ```
    - `<image_path>`: Path to the image for disease prediction.