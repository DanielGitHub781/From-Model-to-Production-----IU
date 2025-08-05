import os
import base64
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime

# Constants
API_URL = "http://localhost:5000/predict"
IMAGE_DIR = Path("Data/new_images")                 
PREDICTED_DIR = Path("Data/predicted_images")       
PREDICTIONS_DIR = Path("Data/predictions")
BATCH_SIZE = 32

# Convert image to base64
def encode_image(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Load images and split into batches
def get_image_batches(folder_path):
    image_paths = list(folder_path.glob('*.jpg')) 
    batches = [image_paths[i:i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    return batches

# Send batch to API
def predict_batch(image_paths):
    base64_images = [encode_image(p) for p in image_paths]
    response = requests.post(API_URL, json={"images": base64_images})
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        return list(zip(image_paths, predictions))
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Move image to predicted folder
def move_image(img_path, predicted_label):
    dest_dir = PREDICTED_DIR / predicted_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), dest_dir / img_path.name)

# Main runner
def run_nightly_batch():
    all_batches = get_image_batches(IMAGE_DIR)
    all_results = []

    for batch in all_batches:
        predictions = predict_batch(batch)
        for img_path, pred_dict in predictions:
            predicted_label = max(pred_dict, key=pred_dict.get)  
            move_image(img_path, predicted_label)
            all_results.append({
                "file": str(img_path.name),
                "predicted_label": predicted_label,
                "probabilities": pred_dict
            })

    # Save results to timestamped JSON file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = PREDICTIONS_DIR / f"nightly_predictions_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Processed {len(all_results)} images.")

if __name__ == "__main__":
    run_nightly_batch()
