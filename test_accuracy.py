import os
import random
from collections import defaultdict
from model import predict

IMAGE_DIR = "dataset/colored_images"
DIAGNOSIS_DIRS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
CLASS_MAPPING = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferate_DR": "Proliferative DR"
}

with open("test_out.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Class (True Condition)':<20} | {'Total Tested'} | {'True Positives'} | {'Accuracy'}\n")
    f.write("-" * 65 + "\n")
    
    for d in DIAGNOSIS_DIRS:
        path = os.path.join(IMAGE_DIR, d)
        true_label = CLASS_MAPPING[d]
        if os.path.exists(path):
            images = os.listdir(path)
            if len(images) > 0:
                samples = random.sample(images, min(20, len(images)))
                true_positives = 0
                for img in samples:
                    img_path = os.path.join(path, img)
                    value, label, confidence = predict(img_path)
                    if label == true_label:
                        true_positives += 1
                
                acc = (true_positives / len(samples)) * 100
                f.write(f"{true_label:<20} | {len(samples):<12} | {true_positives:<14} | {acc:.2f}%\n")
    
    f.write("\nOverall testing finished.\n")
