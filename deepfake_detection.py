# ===================== #
# Deeplearning models #
# ===================== #
# Author:   Amjad Osman,  Simon Lim , Aania Rafi

# Date:     18/4/2024

# How to run:   python combined.py --prototxt [path_to_prototxt] --model [path_to_res10_300x300_ssd_iter_140000.caffemodel] --data-dir [path_to_data_directory] --run-all

# ================= #



import argparse
import os
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from PIL import Image
import imagehash
import cv2
import os
import shutil
import numpy as np

# Sets up argument parser and command line options
def get_args():
    parser = argparse.ArgumentParser(description="Run the image processing and ML model training tasks.")
    parser.add_argument('--prototxt', required=True, help='Path to the Caffe deploy prototxt file.')
    parser.add_argument('--model', required=True, help='Path to the Caffe pre-trained model file.')
    parser.add_argument('--data-dir', required=True, help='Directory with training/validation/test data.')
    parser.add_argument('--run-all', action='store_true', help='Run all tasks sequentially.')
    return parser.parse_args()

# Main function to orchestrate data processing and model training/testing
def main():
    args = get_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    face_net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    if args.run_all:
        print(f"Processing all tasks for the data directory: {args.data_dir}")
        check_corrupt_images(args.data_dir)
        remove_duplicates(args.data_dir)
        detect_and_drop_no_faces(args.data_dir)
        run_model(args.data_dir, 'Xception')
        run_model(args.data_dir, 'ResNet50')
    else:
        print("Please specify --run-all to run all tasks or review your command.")

# def deep_learning_face_detection(image_path, net, confidence_threshold=0.5):
#     image = cv2.imread(image_path)
#     (h, w) = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     faces = []
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > confidence_threshold:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             faces.append((startX, startY, endX, endY))
#     return faces

# def save_cropped_faces(image_path, faces, save_dir):
#     image = cv2.imread(image_path)
#     for i, (x, y, w, h) in enumerate(faces):
#         cropped_face = image[y:y+h, x:x+w]
#         face_file_name = os.path.splitext(os.path.basename(image_path))[0] + f'_face{i}.jpg'
#         cv2.imwrite(os.path.join(save_dir, face_file_name), cropped_face)

# Removes duplicate images from the specified directory
def remove_duplicates(directory):
    print("Removing near duplicates...")
    hashes = {}
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(subdir, file)
                with Image.open(filepath) as img:
                    hash = imagehash.average_hash(img)
                    if hash in hashes:
                        os.remove(filepath)
                        print(f"Removed duplicate: {filepath}")
                        count += 1
                    else:
                        hashes[hash] = filepath
    print(f"Total duplicates removed: {count}")

# Checks for and removes corrupt images in the specified directory
def check_corrupt_images(directory):
    print("Checking for corrupt images...")
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except (IOError, SyntaxError):
                print(f"Corrupt image found and removed: {filepath}")
                os.remove(filepath)
                count += 1
    print(f"Total corrupt images removed: {count}")

# Detects and removes images without faces
def detect_and_drop_no_faces(directory):
    print("Detecting and dropping images with no faces...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count_detected = 0
    count_dropped = 0

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(subdir, file)
                img = cv2.imread(filepath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite(filepath, img)
                    count_detected += 1
                else:
                    os.remove(filepath)  # Remove the image if no faces are detected
                    count_dropped += 1

    print(f"Total images with detected faces: {count_detected}")
    print(f"Total images dropped (no faces detected): {count_dropped}")



# Function to plot the feature maps
def plot_feature_maps(activations, num_columns=8):
    for layer_activation in activations:  
        num_features = layer_activation.shape[-1]  
        size = layer_activation.shape[1]  

        # Tiles each filter into a big horizontal grid
        num_rows = num_features // num_columns
        display_grid = np.zeros((size * num_rows, num_columns * size))

        for row in range(num_rows):
            for col in range(num_columns):
                channel_image = layer_activation[0, :, :, row * num_columns + col]
                
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

        
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

 # Configures and runs a deep learning model based on the specified model name
def run_model(data_dir, model_name):
    print(f"Running {model_name} model...")
    target_size = (299, 299) if model_name == 'Xception' else (224, 224)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(*target_size, 3)) if model_name == 'Xception' else ResNet50(weights='imagenet', include_top=False, input_shape=(*target_size, 3))
    
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")
    val_path = os.path.join(data_dir, "validation")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20 if model_name == 'Xception' else 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(train_path, target_size=target_size, batch_size=32, class_mode='binary')
    test_set = test_datagen.flow_from_directory(test_path, target_size=target_size, batch_size=32, class_mode='binary', shuffle=False)
    val_set = val_datagen.flow_from_directory(val_path, target_size=target_size, batch_size=32, class_mode='binary')

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.02))(x)
    x = Dropout(0.7)(x)
    x = GaussianNoise(0.1)(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.02))(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001)

    history = model.fit(training_set, epochs=10, validation_data=val_set, callbacks=[early_stopping, reduce_lr])

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

# Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Model evaluation
    evaluation = model.evaluate(test_set)
    print(f"Test Loss: {evaluation[0]:.4f}, Test Accuracy: {evaluation[1]:.4f}")

    # Predictions to create confusion matrix, classification report, ROC curve
    test_pred = model.predict(test_set)
    test_pred_classes = np.round(test_pred).astype(int)
    test_true_classes = test_set.classes

    # Confusion Matrix
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(test_true_classes, test_pred_classes, target_names=["Real", "Fake"])
    print("Classification Report:\n", report)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(test_true_classes, test_pred)
    roc_auc = roc_auc_score(test_true_classes, test_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()