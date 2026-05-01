import os
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Optional imports for local testing without tensorflow
try:
    import tensorflow as tf
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not installed. App will run strictly in Mock Mode.")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "models/crop_disease_model.h5"
IMG_SIZE = (224, 224)

# Disease Info Database
DISEASE_INFO = {
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "type": "Fungal Disease",
        "description": "A common tomato disease caused by the fungus Alternaria solani. It appears as dark, concentric rings on older leaves.",
        "cure": "Remove affected leaves immediately. Apply copper-based fungicides or chlorothalonil. Ensure proper spacing for airflow and practice crop rotation."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "type": "Water Mold Disease",
        "description": "A devastating disease caused by Phytophthora infestans. It presents as water-soaked spots on leaves that quickly turn brown and papery.",
        "cure": "Destroy infected plants to prevent spread. Apply fungicides containing chlorothalonil, copper, or mancozeb. Avoid overhead watering."
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "type": "Fungal Disease",
        "description": "Caused by Alternaria solani, characterized by dark brown to black spots with concentric rings on the lower leaves.",
        "cure": "Use certified disease-free seeds. Apply appropriate fungicides when symptoms first appear. Maintain proper plant nutrition and manage irrigation."
    },
    "Potato___healthy": {
        "name": "Healthy Potato Plant",
        "type": "Healthy",
        "description": "Your plant looks healthy and shows no visible signs of the targeted diseases.",
        "cure": "No action needed. Continue providing adequate water, sunlight, and proper nutrition."
    }
}

# The classes exactly as they appear in the dataset folder structure
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

# Try to load the model globally to avoid loading it on every request
model = None
if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model not loaded. App will run in MOCK mode.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(time.time())}_{filename}")
        file.save(filepath)
        
        # If model is not loaded, return mock data for demonstration purposes
        if model is None:
            # Mock prediction (returns a random class)
            import random
            predicted_class = random.choice(CLASS_NAMES)
            confidence = round(random.uniform(75.0, 99.9), 2)
            info = DISEASE_INFO[predicted_class]
            
            return jsonify({
                'success': True,
                'prediction': info['name'],
                'class_id': predicted_class,
                'confidence': f"{confidence}%",
                'type': info['type'],
                'description': info['description'],
                'cure': info['cure'],
                'mock_mode': True
            })
            
        try:
            # Real Prediction
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, 0)  # Batch axis
            
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[class_idx]
            confidence = float(predictions[0][class_idx] * 100)
            
            info = DISEASE_INFO[predicted_class]
            
            return jsonify({
                'success': True,
                'prediction': info['name'],
                'class_id': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'type': info['type'],
                'description': info['description'],
                'cure': info['cure'],
                'mock_mode': False
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
if __name__ == '__main__':
    app.run(debug=True, port=5000)
