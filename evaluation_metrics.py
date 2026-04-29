import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from model import predict

def resolve_image_path(image_id: str, diagnosis: int) -> str:
    DIAGNOSIS_DIR_MAP = {
        0: "No_DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferate_DR",
    }
    return os.path.join("dataset", "colored_images", DIAGNOSIS_DIR_MAP[diagnosis], f"{image_id}.png")

def evaluate():
    TRAIN_CSV = os.path.join("dataset", "train.csv")
    df = pd.read_csv(TRAIN_CSV)
    
    # Filter valid rows
    exists = df.apply(
        lambda row: os.path.exists(resolve_image_path(str(row["id_code"]), int(row["diagnosis"]))),
        axis=1,
    )
    df = df[exists].reset_index(drop=True)
    
    # Get val_df using same seed
    _, val_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["diagnosis"],
        random_state=42,
    )
    
    y_true = []
    y_pred = []
    confidences = []
    
    print(f"Evaluating {len(val_df)} validation images...")
    
    for idx, row in val_df.iterrows():
        img_path = resolve_image_path(str(row["id_code"]), int(row["diagnosis"]))
        true_val = int(row["diagnosis"])
        
        pred_val, pred_label, conf = predict(img_path)
        
        y_true.append(true_val)
        y_pred.append(pred_val)
        confidences.append(conf)
        
    print("\n--- RESULTS ---")
    conf_matrix = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print(f"Actual Accuracy: {acc*100:.2f}%")
    print(f"Prediction Confidence Accuracy: {avg_conf:.2f}%")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nConfidence (Confusion) Matrix:")
    print(conf_matrix)
    
if __name__ == "__main__":
    evaluate()
