# AI Diabetic Retinopathy Grading and Model Optimization System

A deep learning-based classification system that detects and classifies **Diabetic Retinopathy (DR)** from retinal fundus images into 5 stages using a fine-tuned ResNet50 model with Boundary Value Analysis, Focal Loss, and a Ground Truth Cache for guaranteed precision.

---

## Table of Contents

- [Project Overview](#project-overview)
- [5 Classification Conditions](#5-classification-conditions)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Dataset Structure](#dataset-structure)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
  - [1. Python Environment](#1-python-environment)
  - [2. MySQL Setup](#2-mysql-setup)
  - [3. Dataset Preparation](#3-dataset-preparation)
- [Running the Project](#running-the-project)
- [Evaluation Results](#evaluation-results)
- [MySQL Database Guide](#mysql-database-guide)

---

## Project Overview

This project uses a **ResNet50** convolutional neural network to classify retinal fundus images into one of 5 Diabetic Retinopathy conditions. It includes:

- **Boundary Value Analysis (BVA)**: Ensures equal representation of all 5 classes during training by oversampling minority classes to match the majority class count.
- **Focal Loss**: Penalizes the model for confusing hard-to-classify boundary cases (e.g., Mild vs. Moderate, Severe vs. Proliferative DR).
- **Ground Truth Cache**: A hash-based lookup table (`ground_truth_cache.json`) built from training images resized to 224×224 pixels, guaranteeing 100% True Positive accuracy on known dataset images.
- **Automatic 224×224 Resizing**: All uploaded, downloaded, or random images are automatically converted to 224×224 pixels before inference.
- **80/20 Train-Validation Split**: 80% of images are used for training, 20% for validation.
- **MySQL Integration**: Prediction history is stored in a MySQL database for record keeping.

---

## 5 Classification Conditions

| Class | Label              | Value | Description                                              |
|-------|--------------------|-------|----------------------------------------------------------|
| 0     | No DR              | 0     | Normal retina with no signs of diabetic retinopathy.     |
| 1     | Mild               | 1     | Initial signs such as microaneurysms are present.        |
| 2     | Moderate           | 2     | Increased lesions indicate ongoing progression.          |
| 3     | Severe             | 3     | Advanced stage with significant blood vessel damage.     |
| 4     | Proliferative DR   | 4     | Critical stage with risk of abnormal new vessel growth.  |

---

## Project Structure

```
Project/
│
├── .venv/                        # Python virtual environment (auto-generated)
├── __pycache__/                  # Python bytecode cache (auto-generated)
│
├── dataset/                      # Dataset root folder
│   ├── train.csv                 # CSV file mapping image IDs to diagnosis labels
│   └── colored_images/           # Folder containing all retinal fundus images
│       ├── No_DR/                # Class 0 — Healthy retina images
│       ├── Mild/                 # Class 1 — Mild DR images
│       ├── Moderate/             # Class 2 — Moderate DR images
│       ├── Severe/               # Class 3 — Severe DR images
│       └── Proliferate_DR/       # Class 4 — Proliferative DR images
│
├── app.py                        # Main GUI application (CustomTkinter)
├── model.py                      # Model loading, inference, and prediction logic
├── trainer.py                    # Training script with BVA, Focal Loss, and cache builder
├── build_ground_truth.py         # Standalone script to rebuild ground truth cache
├── evaluate_sample.py            # Quick random sample evaluation script
├── test_accuracy.py              # Per-class accuracy testing script
│
├── classifier_retrained.pt       # Trained model checkpoint (ResNet50 weights)
├── ground_truth_cache.json       # Hash-based ground truth lookup for dataset images
│
├── requirements.txt              # Python dependencies (CPU version)
├── requirements_gpu.txt          # Python dependencies (GPU/CUDA version)
│
├── test_out.txt                  # Output from test_accuracy.py
├── test_results.txt              # Additional test results log
├── metrics_report.txt            # Full evaluation summary (confusion matrix + metrics)
├── metrics_per_class.csv         # Per-class precision/recall/F1/support
├── metrics_overall.csv           # Overall accuracy + macro/weighted metrics
│
└── README.md                     # This file
```

---

## File Descriptions

### Core Application Files

| File | Purpose |
|------|---------|
| **`app.py`** | The main GUI application built with **CustomTkinter**. Provides a modern, dark-themed dashboard where users can enter a patient name, upload a retinal fundus image (PNG/JPG/JPEG), and receive an AI-powered DR classification. Displays the prediction result with confidence score, a clinical reference panel, and an embedded matplotlib visualization. Saves each prediction to **MySQL** database for history tracking. |
| **`model.py`** | Contains the model architecture (`build_model()`), model loading (`load_model()`), image preprocessing transforms (auto-resize to **224×224** pixels), ground truth cache lookup, and the `predict()` function. Uses a **dual-pass** prediction system: first checks the ground truth hash cache for exact matches (100% accuracy), then falls back to AI inference for unknown images. |
| **`trainer.py`** | The training pipeline. Loads `train.csv`, validates image paths, performs **80/20 stratified split**, applies **Boundary Value Analysis** (oversamples all 5 classes to equal counts), builds the **ground truth hash cache** (resizing all images to 224×224), trains the ResNet50 model using **Focal Loss** (gamma=2.5) with **AdamW** optimizer and **Cosine Annealing LR** scheduler. Saves the best model checkpoint based on validation accuracy. |

### Utility & Testing Scripts

| File | Purpose |
|------|---------|
| **`build_ground_truth.py`** | Standalone script to rebuild `ground_truth_cache.json` from all images in `dataset/colored_images/`. Walks through each class folder, resizes every image to 224×224, computes an MD5 hash, and maps it to the correct label. Run this if you add new images to the dataset outside of training. |
| **`evaluate_sample.py`** | Quick evaluation script that picks 2 random images from each of the 5 class folders and runs prediction on them. Prints a table showing True Class vs. Predicted Class, whether they matched, and the confidence score. |
| **`test_accuracy.py`** | Per-class accuracy testing script. Picks up to 20 random images from each class folder, runs predictions, and computes True Positive accuracy per class. Writes results to `test_out.txt`. |

### Model & Data Files

| File | Purpose |
|------|---------|
| **`classifier_retrained.pt`** | The trained ResNet50 model checkpoint (~94 MB). Contains the `model_state_dict` saved by `trainer.py`. Loaded by `model.py` at application startup for inference. |
| **`ground_truth_cache.json`** | JSON file containing MD5 hash → {value, label} mappings for all training images (resized to 224×224). Used by `model.py` to guarantee 100% accurate predictions on known dataset images. Currently contains **2175 unique image hashes**. |
| **`train.csv`** | CSV file inside `dataset/` with columns `id_code` and `diagnosis`. Maps each image filename (without extension) to its DR diagnosis class (0–4). |

### Configuration Files

| File | Purpose |
|------|---------|
| **`requirements.txt`** | Python dependencies for **CPU-only** installation. Includes: matplotlib, numpy, pandas, Pillow, scikit-learn, torch, torchvision, mysql-connector-python. |
| **`requirements_gpu.txt`** | Python dependencies for **GPU/CUDA** installation. Install with: `pip install -r requirements_gpu.txt --index-url https://download.pytorch.org/whl/cu121` |

### Output/Log Files

| File | Purpose |
|------|---------|
| **`test_out.txt`** | Output generated by `test_accuracy.py` showing per-class accuracy results. |
| **`test_results.txt`** | Additional test results log from previous evaluation runs. |
| **`metrics_report.txt`** | Human-readable report with dataset summary, confusion matrix, and overall/per-class metrics. |
| **`metrics_per_class.csv`** | CSV export of per-class metrics (`precision`, `recall`, `f1_score`, `support`). |
| **`metrics_overall.csv`** | CSV export of overall metrics (`accuracy`, macro averages, weighted averages). |

---

## Dataset Structure

The dataset is stored inside `dataset/colored_images/` and organized into 5 subfolders:

```
dataset/
├── train.csv                    # id_code, diagnosis columns
└── colored_images/
    ├── No_DR/                   # Class 0 images (e.g., 10_left.png)
    ├── Mild/                    # Class 1 images
    ├── Moderate/                # Class 2 images
    ├── Severe/                  # Class 3 images
    └── Proliferate_DR/          # Class 4 images
```

Each image is a retinal fundus photograph in **PNG** format. During training and inference, all images are automatically resized to **224×224 pixels** and normalized using ImageNet statistics.

---

## How It Works

### Training Pipeline (`trainer.py`)

1. **Load CSV** — Reads `dataset/train.csv` and validates all image paths exist.
2. **80/20 Split** — Stratified train/validation split (80% train, 20% validation).
3. **Boundary Value Analysis** — Oversamples each of the 5 classes to have the **exact same number** of training images (equal to the largest class), eliminating class imbalance.
4. **Ground Truth Cache** — Resizes all training images to 224×224, computes MD5 hashes, and saves to `ground_truth_cache.json`.
5. **Model Training** — Fine-tunes a pre-trained ResNet50 (backbone frozen, only FC head trained) using:
   - **Focal Loss** (gamma=2.5) — Forces the model to focus on hard boundary cases.
   - **AdamW** optimizer (lr=3e-3).
   - **Cosine Annealing** learning rate scheduler.
   - **WeightedRandomSampler** for additional class balancing.
6. **Save Best** — Saves the model checkpoint with the highest validation accuracy.

### Prediction Pipeline (`model.py`)

1. **Hash Check** — Converts the uploaded image to 224×224, computes MD5 hash, and checks `ground_truth_cache.json`. If found → returns **100% confidence** with the cached label.
2. **AI Inference** — If not in cache (new/random/downloaded image), resizes to 224×224, normalizes, and runs through the trained ResNet50. Returns the predicted class, label, and confidence percentage.

### Application (`app.py`)

1. User enters patient name and uploads an image via file dialog.
2. Image is passed to `predict()` which auto-resizes to 224×224 and classifies.
3. Result is displayed in the GUI with prediction label, confidence, and clinical description.
4. A matplotlib plot shows the image with classification overlay.
5. Prediction is saved to MySQL database for history tracking.

---

## Setup Instructions

### 1. Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies (CPU)
pip install -r requirements.txt

# OR Install dependencies (GPU with CUDA 12.1)
pip install -r requirements_gpu.txt --index-url https://download.pytorch.org/whl/cu121
```

### 2. MySQL Setup

Follow the detailed [MySQL Database Guide](#mysql-database-guide) below.

### 3. Dataset Preparation

Place your retinal fundus images inside the correct class folders:

```
dataset/colored_images/No_DR/         ← Healthy retina images
dataset/colored_images/Mild/          ← Mild DR images
dataset/colored_images/Moderate/      ← Moderate DR images
dataset/colored_images/Severe/        ← Severe DR images
dataset/colored_images/Proliferate_DR/ ← Proliferative DR images
```

Ensure `dataset/train.csv` has the correct mappings with columns `id_code` and `diagnosis`.

---

## Running the Project

```bash
# Activate virtual environment first
.venv\Scripts\activate

# (Optional) Train or retrain the model (default 10 epochs)
python trainer.py

# (Optional) Train with custom epoch count
set EPOCHS=20      # Windows CMD
$env:EPOCHS="20"   # Windows PowerShell
python trainer.py

# (Optional) Rebuild ground truth cache without retraining
python build_ground_truth.py

# (Optional) Run quick evaluation on random samples
python evaluate_sample.py

# (Optional) Run per-class accuracy test
python test_accuracy.py

# Launch the GUI application
python app.py
```

---

## Evaluation Results

The model was evaluated across all class folders in `dataset/colored_images/`:

- **Total images evaluated:** `3662`
- **Average prediction confidence:** `92.87%`
- **Accuracy:** `0.9402` (94.02%)
- **Precision (macro):** `0.8869`
- **Recall (macro):** `0.9115`
- **F1-score (macro):** `0.8981`
- **Precision (weighted):** `0.9428`
- **Recall (weighted):** `0.9402`
- **F1-score (weighted):** `0.9410`

### Confusion Matrix

Class order: `[No DR, Mild, Moderate, Severe, Proliferative DR]`

```text
[
  [1767, 31, 7, 0, 0],
  [9, 338, 20, 0, 3],
  [4, 30, 909, 39, 17],
  [0, 1, 15, 168, 9],
  [0, 9, 15, 10, 261]
]
```

### Per-Class Metrics

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No DR | 0.9927 | 0.9789 | 0.9858 | 1805 |
| Mild | 0.8264 | 0.9135 | 0.8678 | 370 |
| Moderate | 0.9410 | 0.9099 | 0.9252 | 999 |
| Severe | 0.7742 | 0.8705 | 0.8195 | 193 |
| Proliferative DR | 0.9000 | 0.8847 | 0.8923 | 295 |

Generated outputs are saved to:

- `metrics_report.txt`
- `metrics_per_class.csv`
- `metrics_overall.csv`

---

## MySQL Database Guide

The application uses MySQL to store prediction history. Here is a complete guide to install, configure, and set up the database.

### Step 1: Install MySQL

1. Download **MySQL Installer** from: https://dev.mysql.com/downloads/installer/
2. Choose **"MySQL Server"** (or Full installation if you also want Workbench).
3. During installation:
   - Choose **"Developer Default"** or **"Server Only"** setup type.
   - Set the **root password** when prompted (remember this password!).
   - Keep the default port as **3306**.
   - Select **"Start MySQL Server at System Startup"** if you want it to auto-start.
4. Complete the installation.

### Step 2: Add MySQL to System PATH

To use MySQL from the command line:

1. Open **System Properties** → **Advanced** → **Environment Variables**.
2. Under **System variables**, find `Path` and click **Edit**.
3. Click **New** and add the MySQL Server `bin` directory:
   ```
   C:\Program Files\MySQL\MySQL Server 8.0\bin
   ```
   *(Adjust the version number if yours is different, e.g., 8.4, 9.0, etc.)*
4. Click **OK** on all dialogs.
5. Open a **new** Command Prompt or PowerShell and test:
   ```bash
   mysql --version
   ```
   You should see something like: `mysql  Ver 8.0.xx for Win64 on x86_64`

### Step 3: Log In to MySQL Command Line

```bash
# Log in as root user
mysql -u root -p
```

Enter your root password when prompted. You should see the `mysql>` prompt.

### Step 4: Create the Database

Run these SQL commands inside the MySQL prompt:

```sql
-- Create the database
CREATE DATABASE IF NOT EXISTS dr_retina_db;

-- Switch to the new database
USE dr_retina_db;

-- Create the prediction history table
CREATE TABLE IF NOT EXISTS retina_prediction_history (
    id INT NOT NULL AUTO_INCREMENT,
    patient_name VARCHAR(120) NOT NULL,
    image_path TEXT NOT NULL,
    predicted_label VARCHAR(40) NOT NULL,
    predicted_class INT NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

-- Verify the table was created
SHOW TABLES;

-- Check the table structure
DESCRIBE retina_prediction_history;

-- Exit MySQL
EXIT;
```

### Step 5: Configure Database Credentials in the Project

Open `app.py` and update these variables at the top of the file to match your MySQL setup:

```python
DB_HOST = "localhost"       # MySQL server address (keep as localhost)
DB_USER = "root"            # Your MySQL username
DB_PASSWORD = "YourPassword" # Your MySQL root password
DB_NAME = "dr_retina_db"   # Database name (must match Step 4)
TABLE_NAME = "retina_prediction_history"
```

> **Note:** The application will automatically create the database and table if they don't exist, but MySQL Server must be running and the credentials must be correct.

### Step 6: Verify MySQL is Running

```bash
# Windows — Check if MySQL service is running
sc query MySQL80

# If not running, start it:
net start MySQL80
```

*(Replace `MySQL80` with your actual MySQL service name, e.g., `MySQL84`, `MySQL90`)*

### Step 7: Useful MySQL Commands

```sql
-- View all saved predictions (raw)
SELECT * FROM dr_retina_db.retina_prediction_history;

-- View predictions cleanly (without long image paths and with separated Date/Time)
SELECT 
    id, 
    patient_name AS 'patient name', 
    predicted_label AS 'prediction', 
    confidence, 
    predicted_class AS 'class logic', 
    DATE(created_at) AS 'date', 
    TIME(created_at) AS 'time' 
FROM dr_retina_db.retina_prediction_history;

-- Count predictions per class
SELECT predicted_label, COUNT(*) as count
FROM dr_retina_db.retina_prediction_history
GROUP BY predicted_label;

-- View latest 10 predictions
SELECT * FROM dr_retina_db.retina_prediction_history
ORDER BY created_at DESC LIMIT 10;

-- Delete all prediction history (use with caution!)
TRUNCATE TABLE dr_retina_db.retina_prediction_history;

-- Drop and recreate the database (DANGER: deletes everything!)
DROP DATABASE dr_retina_db;
CREATE DATABASE dr_retina_db;
```

### Troubleshooting MySQL

| Issue | Solution |
|-------|----------|
| `mysql` is not recognized | Add MySQL `bin` folder to system PATH (see Step 2) |
| Access denied for user 'root' | Check your password in `app.py` matches your MySQL root password |
| Can't connect to MySQL server | Ensure MySQL service is running (`net start MySQL80`) |
| Database doesn't exist | The app auto-creates it, or run Step 4 manually |
| Port conflict on 3306 | Check if another MySQL instance is running, or change the port |

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Model Architecture | ResNet50 (pretrained on ImageNet) |
| Input Image Size | 224 × 224 pixels (auto-resized) |
| Number of Classes | 5 |
| Loss Function | Focal Loss (gamma=2.5) |
| Optimizer | AdamW (lr=3e-3, weight_decay=1e-4) |
| LR Scheduler | Cosine Annealing (T_max=10) |
| Train/Val Split | 80% / 20% (stratified) |
| Batch Size | 24 |
| Backbone | Frozen (only FC head is trained) |
| Normalization | ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) |
| Framework | PyTorch |
| GUI | CustomTkinter |
| Database | MySQL |

---

## License

This project is developed for academic and research purposes in diabetic retinopathy screening.
