import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import base64

# Constants
IMG_SIZE = (150, 150)
NUM_CLASSES = 0

# Load model and class mapping
model = tf.keras.models.load_model('image_classifier_model.keras')
with open('class_indices.json', 'r') as f:
    class_to_index = json.load(f)
index_to_class = {v: k for k, v in class_to_index.items()}
NUM_CLASSES = len(index_to_class)

# Create Flask app
app = Flask(__name__)

# Preprocess image function
def preprocess_image(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400

        images_data = data['images']  # List of base64 strings
        images = [preprocess_image(img) for img in images_data]
        batch = np.stack(images)

        preds = model.predict(batch)

        results = []
        for prob_array in preds:
            result = {
                index_to_class[i]: float(prob)
                for i, prob in enumerate(prob_array)
            }
            results.append(result)

        return jsonify({'predictions': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "Image Classifier API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
