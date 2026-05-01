import os
import shutil
import zipfile
import subprocess
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Config
DATASET_NAME = "emmarex/plantdisease"
ZIP_FILE = "plantdisease.zip"
BASE_EXTRACT_DIR = "temp_plantdisease"
FINAL_DATA_DIR = "data"
MODEL_SAVE_PATH = "models/crop_disease_model.h5"

TARGET_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy"
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

def setup_kaggle_and_download():
    """
    Downloads the dataset from Kaggle.
    Requires kaggle.json to be configured locally or uploaded in Colab.
    """
    if os.path.exists(FINAL_DATA_DIR) and len(os.listdir(FINAL_DATA_DIR)) > 0:
        print("Data directory already exists and is not empty. Skipping download.")
        return

    print("Downloading dataset from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_NAME], check=True)
        print("Download complete.")
    except Exception as e:
        print("Failed to download dataset. Ensure Kaggle API is set up properly.")
        print(e)
        return

    print("Extracting and filtering data...")
    if not os.path.exists(BASE_EXTRACT_DIR):
        os.makedirs(BASE_EXTRACT_DIR)

    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(BASE_EXTRACT_DIR)

    # The dataset structure is typically temp_plantdisease/PlantVillage/... or temp_plantdisease/plantvillage/...
    # Let's search for the target classes
    if not os.path.exists(FINAL_DATA_DIR):
        os.makedirs(FINAL_DATA_DIR)

    found_classes = 0
    for root, dirs, files in os.walk(BASE_EXTRACT_DIR):
        for dir_name in dirs:
            if dir_name in TARGET_CLASSES:
                src_path = os.path.join(root, dir_name)
                dest_path = os.path.join(FINAL_DATA_DIR, dir_name)
                if not os.path.exists(dest_path):
                    shutil.move(src_path, dest_path)
                    print(f"Moved {dir_name} to final data directory.")
                found_classes += 1
    
    if found_classes == 0:
        print("Warning: Target classes not found in the extracted archive.")

    # Cleanup
    print("Cleaning up temporary files to save space...")
    shutil.rmtree(BASE_EXTRACT_DIR, ignore_errors=True)
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
    print("Cleanup complete.")

def load_and_preprocess_data():
    """
    Loads data from directory, splits into train/val/test, and normalizes.
    70% train, 15% validation, 15% test.
    """
    print("Loading data...")
    
    # We use tf.keras.preprocessing.image_dataset_from_directory
    # We will split into 70% train, 30% val_test
    train_dataset = image_dataset_from_directory(
        FINAL_DATA_DIR,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_test_dataset = image_dataset_from_directory(
        FINAL_DATA_DIR,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    # Split val_test into 50% validation (15% total) and 50% test (15% total)
    val_test_batches = tf.data.experimental.cardinality(val_test_dataset)
    val_dataset = val_test_dataset.take(val_test_batches // 2)
    test_dataset = val_test_dataset.skip(val_test_batches // 2)
    
    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset, train_dataset.class_names

def build_model(num_classes):
    """
    Builds MobileNetV2 with Transfer Learning.
    """
    print("Building model...")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, train_data, val_data):
    """
    Trains the model and saves the best one.
    """
    print("Starting training...")
    if not os.path.exists("models"):
        os.makedirs("models")
        
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    return history

def evaluate_model(model, test_data, class_names):
    """
    Evaluates the model and prints classification report & confusion matrix.
    """
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Get predictions for confusion matrix
    y_true = []
    y_pred = []
    for x, y in test_data:
        preds = model.predict(x)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    print("Saved confusion matrix plot to models/confusion_matrix.png")

def predict_disease(image_path):
    """
    Predicts the disease for a single image.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH}. Please train it first.")
        return
        
    print(f"Predicting for image: {image_path}")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    # Image loading and preprocessing
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    
    # Target classes matching the categorical setup
    # The order of class_names depends on os.listdir but image_dataset_from_directory sorts alphanumerically.
    # We should reconstruct class_names similarly.
    classes_sorted = sorted(os.listdir(FINAL_DATA_DIR))
    
    predicted_class = classes_sorted[class_idx]
    confidence = predictions[0][class_idx] * 100
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    return predicted_class, confidence

if __name__ == "__main__":
    print("=== Crop Disease Detection System ===")
    # 1. Download and preprocess dataset
    setup_kaggle_and_download()
    
    # 2. Check if data exists
    if not os.path.exists(FINAL_DATA_DIR) or len(os.listdir(FINAL_DATA_DIR)) == 0:
        print("Data is missing. Exiting.")
        exit(1)
        
    # 3. Load data
    # Note: image_dataset_from_directory class_names property is accessible before the map.
    # We will instantiate it slightly differently to grab class_names easily.
    train_ds_raw = image_dataset_from_directory(
        FINAL_DATA_DIR,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    class_names = train_ds_raw.class_names
    print(f"Class names found: {class_names}")
    
    # Load fully processed data
    train_data, val_data, test_data, _ = load_and_preprocess_data()
    
    # 4. Build and train model
    model = build_model(num_classes=len(class_names))
    train_model(model, train_data, val_data)
    
    # 5. Evaluate model (using the saved best model)
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    evaluate_model(best_model, test_data, class_names)
    
    # Example prediction usage
    # predict_disease("data/Potato___healthy/some_image.jpg")
