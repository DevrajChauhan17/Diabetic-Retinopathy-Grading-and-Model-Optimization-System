import os
import random
from model import predict

IMAGE_DIR = "dataset/colored_images"
DIAGNOSIS_DIRS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
# To match model.py class names which are: ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_MAPPING = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferate_DR": "Proliferative DR"
}

print(f"{'True Class':<18} | {'Predicted Class':<18} | {'Matched?':<8} | {'Confidence'}")
print("-" * 70)

for d in DIAGNOSIS_DIRS:
    path = os.path.join(IMAGE_DIR, d)
    if os.path.exists(path):
        images = os.listdir(path)
        if len(images) > 0:
            # pick 2 random images
            samples = random.sample(images, min(2, len(images)))
            for img in samples:
                img_path = os.path.join(path, img)
                value, label, confidence = predict(img_path)
                true_label = CLASS_MAPPING[d]
                matched = "Yes" if label == true_label else "No"
                print(f"{true_label:<18} | {label:<18} | {matched:<8} | {confidence:.2f}%")
