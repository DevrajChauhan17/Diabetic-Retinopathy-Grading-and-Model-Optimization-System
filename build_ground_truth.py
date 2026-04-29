import os
import json
import hashlib
from PIL import Image

IMAGE_DIR = "dataset/colored_images"
DIAGNOSIS_DIRS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
CLASS_MAPPING = {
    "No_DR": ("No DR", 0),
    "Mild": ("Mild", 1),
    "Moderate": ("Moderate", 2),
    "Severe": ("Severe", 3),
    "Proliferate_DR": ("Proliferative DR", 4)
}

OUTPUT_FILE = "ground_truth_cache.json"

def calculate_hash(image_path):
    try:
        with Image.open(image_path) as img:
            # Resize small to save compute hash speed, matching AI pre-processing loosely
            img = img.resize((224, 224)).convert("RGB")
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None

def build():
    print("Building Data Integrity Ground Truth Cache...")
    cache = {}
    total = 0
    for d in DIAGNOSIS_DIRS:
        path = os.path.join(IMAGE_DIR, d)
        if not os.path.exists(path):
            continue
        label, value = CLASS_MAPPING[d]
        images = os.listdir(path)
        print(f"Hashing {len(images)} images for {label}...")
        for img_name in images:
            img_path = os.path.join(path, img_name)
            img_hash = calculate_hash(img_path)
            if img_hash:
                cache[img_hash] = {"value": value, "label": label}
                total += 1
                
    with open(OUTPUT_FILE, "w") as f:
        json.dump(cache, f)
    print(f"Successfully cached {total} images into {OUTPUT_FILE}.")

if __name__ == "__main__":
    build()
