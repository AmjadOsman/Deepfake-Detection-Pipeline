
# Deepfake Detection Pipeline

This project is aimed at detecting deepfake images using state-of-the-art pre-trained models such as **Xception** and **ResNet50**. The script performs image preprocessing (such as removing corrupt images, near-duplicate images, and images without faces) and trains a deep learning model to classify images as real or deepfake.

## Features

- **Corrupt Image Detection**: Automatically identifies and removes any corrupt images from the dataset.
- **Duplicate Image Removal**: Detects and removes near-duplicate images using image hashing.
- **Face Detection**: Removes images that do not contain a detectable face, using OpenCV's Haar cascade.
- **Deepfake Classification Models**: Transfer learning using pre-trained **Xception** and **ResNet50** models, fine-tuned to detect deepfake images.
- **Training Visualization**: Provides accuracy and loss plots for both the training and validation datasets.

---

## Dependencies

To run the script, the following Python libraries are required:

- `tensorflow` (Deep Learning framework)
- `opencv-python` (OpenCV for image processing and face detection)
- `imagehash` (For detecting duplicate images using perceptual hashing)
- `Pillow` (For image manipulation and loading)
- `numpy` (For numerical computations)
- `matplotlib` (For plotting training results)
- `seaborn` (For enhanced visualizations)
- `tqdm` (Progress bars for long-running tasks)
- `scikit-learn` (For evaluation metrics like ROC curves and classification reports)
- `argparse` (Command-line argument parsing)

Install the dependencies via pip:

```bash
pip install tensorflow opencv-python imagehash Pillow numpy matplotlib seaborn tqdm scikit-learn
```

---

## Script Usage

The script is designed to handle several preprocessing and training tasks. It allows you to check for corrupt images, remove duplicates, detect images without faces, and train models to detect deepfakes. You can also run all tasks sequentially using the `--run-all` flag.

### Command-Line Arguments

- **Required Arguments**:
  - `--prototxt`: Path to the Caffe deploy `.prototxt` file used for face detection.
  - `--model`: Path to the Caffe pre-trained `.caffemodel` file for face detection.
  - `--data-dir`: Path to the dataset directory containing `train`, `test`, and `validation` folders.

- **Optional Flags**:
  - `--run-all`: Run all the image processing and model training tasks sequentially.
  - `--check-corrupt`: Check for and remove corrupt images.
  - `--remove-duplicates`: Remove duplicate images based on perceptual hashing.
  - `--detect-no-faces`: Detect and remove images without faces.
  - `--run-xception`: Train the Xception model for deepfake detection.
  - `--run-resnet50`: Train the ResNet50 model for deepfake detection.

### Example Command

To run the script for all tasks in sequence, use the following command:

```bash
python deepfake_detection.py --prototxt path/to/deploy.prototxt --model path/to/model.caffemodel --data-dir path/to/dataset --run-all
```

Alternatively, you can run specific tasks by specifying the corresponding flags:

```bash
python deepfake_detection.py --prototxt path/to/deploy.prototxt --model path/to/model.caffemodel --data-dir path/to/dataset --check-corrupt --remove-duplicates
```

---

## Key Functions

### 1. **`check_corrupt_images(directory)`**

This function scans the dataset directory for corrupt images. If any corrupt images are found (i.e., images that cannot be opened), they are removed from the directory.

### 2. **`remove_duplicates(directory)`**

Using perceptual image hashing, this function identifies and removes near-duplicate images in the dataset. If multiple images with the same hash are found, all but one are deleted.

### 3. **`detect_and_drop_no_faces(directory)`**

This function uses OpenCV’s Haar Cascade classifier to detect faces in each image. If no faces are detected in an image, that image is removed from the dataset.

### 4. **`run_model(data_dir, model_name)`**

This function trains either the **Xception** or **ResNet50** model on the image dataset to classify images as real or deepfake. It uses transfer learning, where pre-trained models are fine-tuned for the deepfake detection task.

#### Steps:
1. **Data Augmentation**: Applies random transformations (e.g., rotations, shifts, flips) to increase the diversity of the training data.
2. **Transfer Learning**: The pre-trained models (Xception or ResNet50) are loaded with their top layers removed. New layers are added for binary classification (real vs. deepfake).
3. **Training**: The model is trained on the provided dataset using the Adam optimizer and binary cross-entropy loss. Early stopping and learning rate reduction are used to improve the training process.
4. **Evaluation**: After training, the model is evaluated on the validation dataset. Accuracy and loss curves are generated for analysis.

### 5. **Progress Monitoring**

The `tqdm` library is used to display progress bars for long-running tasks, such as checking corrupt images and removing duplicates, providing a visual representation of task completion.

---

## Directory Structure

The dataset should be structured as follows:

```bash
dataset/
│
├── train/
│   ├── real/
│   └── deepfake/
│
├── validation/
│   ├── real/
│   └── deepfake/
│
└── test/
    ├── real/
    └── deepfake/
```

Each folder (`train`, `validation`, `test`) should contain two subdirectories: `real` (for real images) and `deepfake` (for deepfake images).

---

## Output

- **Corrupt/Duplicate Images**: The script prints the names of corrupt or duplicate images that were removed during preprocessing.
- **Model Training**: The training process outputs the accuracy and loss for each epoch, along with validation performance.
- **Training Visualizations**: After training, the script generates plots showing the accuracy and loss trends over time for both the training and validation datasets.

### Example Plots

The following plots are generated after training:

1. **Accuracy Plot**: Visualizes the model's training and validation accuracy over the epochs.
2. **Loss Plot**: Visualizes the model's training and validation loss over the epochs.

---

## Evaluation Metrics

After training, the model's performance can be evaluated using various metrics, such as:

- **Confusion Matrix**: Provides insight into how many true positives, true negatives, false positives, and false negatives occurred.
- **ROC Curve and AUC Score**: Measures the model's ability to distinguish between real and deepfake images.

---

## Example Output

- **Train Accuracy**: 92.5%
- **Validation Accuracy**: 90.1%
- **ROC-AUC Score**: 0.94

---

## Notes

- **Face Detection**: Ensure that the paths to the Caffe `.prototxt` and `.caffemodel` files are correct. These are required for face detection.
- **GPU Support**: The script is GPU-enabled. If a GPU is available, TensorFlow will use it by default, leading to faster training.
- **Customizations**: You can modify the image augmentation parameters, model architecture, and optimizer settings to suit your specific use case.

