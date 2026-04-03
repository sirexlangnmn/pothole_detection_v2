# AyosBayan ML Module: Automated Detection of Infrastructure Issues

## 1. Project Overview

**AyosBayan** is a transparent and responsive citizen infrastructure reporting system for public safety and welfare in the Philippines.

This repository contains the **Machine Learning Module #1**:
**Automated Detection of Infrastructure Issues for Citizen Reporting Systems**.

### Core Idea
When a citizen uploads a photo of a road problem through the AyosBayan web platform, the ML model **automatically detects** whether the image contains a **pothole** (initial single-class detection).  

The model instantly returns:
- Detection result (pothole / no pothole)
- Confidence score (0.0 – 1.0)
- Bounding box (optional, for visual feedback)

### Classification
- **Current Class**: `pothole` (single-class object detection)

### Key Benefits (for LGUs and the System)
- **Faster verification** – Reports are auto-tagged; administrators see pre-classified issues instantly.
- **Reduces manual checking** – Only low-confidence or suspicious reports need human review (estimated 60-70% reduction in manual image inspection).
- **Detects fake reports** – If no pothole is detected or confidence is below threshold, the report is automatically flagged for manual review.

This module directly supports the Design Science Research goals of the AyosBayan paper by turning raw citizen photos into **actionable, verified data** for local government units (LGUs).

---

## 2. How It Works (High-Level Workflow)

1. Citizen uploads photo
2. Backend sends the photo to the ML inference API.
3. YOLOv8 model processes the image in real-time (< 2 seconds).
4. System receives:
   - `detected: true/false`
   - `confidence: 0.92`
   - `is_potential_fake: false`
5. Report is auto-tagged as "Pothole" with confidence score.
6. LGU dashboard shows the tagged report with visual bounding box overlay.

---

## 3. Technology Stack

| Component              | Technology                          |
|------------------------|-------------------------------------|
| Model                  | YOLOv8 (Ultralytics)                |
| Framework              | PyTorch (via Ultralytics)           |
| Language               | Python 3.10+                        |
| API                    | FastAPI (recommended)               |
| Training               | Google Colab (free GPU)             |
| Dataset                | Roboflow Pothole Detection Dataset + custom citizen data |
| Deployment             | Docker-ready (planned)              |

---

## 4. Project Structure (Recommended)

ayosbayan-ml/
├── ReadMe_v2.md                       ← You are here
├── requirements.txt
├── best.pt                            ← Trained model (not pushed to Git)
├── src/
│   ├── train.py                       ← Training script
│   ├── inference.py                   ← Local testing script
│   ├── api.py                         ← FastAPI endpoint
│   └── utils.py                       ← Helper functions
├── data/                              ← Dataset folder (gitignore)
│   └── pothole_dataset/               ← data.yaml + images + labels
├── runs/                              ← Training results (gitignore)
├── notebooks/                         ← Google Colab notebooks
│   └── 01_train_pothole_yolov8.ipynb
├── tests/
│   └── test_images/                   ← Sample citizen photos for testing
└── docker/                            ← Future deployment files


---

## 5. Installation & Setup

1. Clone the repository (or your AyosBayan main repo)
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt

requirements.txt content (already prepared for you):
ultralytics==8.3.0
fastapi==0.115.0
uvicorn==0.30.0
pillow==10.4.0
python-multipart==0.0.12

---

## 6. Usage

6.1 Training the Model (Google Colab recommended)
See notebooks/01_train_pothole_yolov8.ipynb
6.2 Local Inference (Testing)
Bashpython src/inference.py --image tests/test_images/pothole_example.jpg
6.3 Running the API (Production Backend)
Bashuvicorn src.api:app --reload --port 8000
Endpoint: POST /detect-pothole
Request: Upload a photo file
Response (JSON):
JSON{
  "detected": true,
  "confidence": 0.94,
  "is_potential_fake": false,
  "auto_tag": "pothole",
  "message": "Pothole detected with high confidence"
}

---

## 7. Fake Report Detection Logic

PythonCONFIDENCE_THRESHOLD = 0.5

if not detected or confidence < CONFIDENCE_THRESHOLD:
    # Flag as potential fake / needs manual review

---

## 8. Purpose of This README for LLM AI Agents

This file serves as the official guideline for any LLM AI agent (Grok, Claude, GPT, etc.) helping with this project.
When you read this README, you must:

Always keep the three core benefits in mind: Faster verification, Reduced manual checking, Detect fake reports.
Use YOLOv8 as the default model unless explicitly asked otherwise.
Keep code simple, well-commented, and beginner-friendly (the developer only knows Python).
Follow the exact folder structure and naming conventions above.
Prioritize real-time inference (< 2 seconds) and clear confidence scoring.
Never suggest complex frameworks outside the listed tech stack.

Project Goal Reminder:
Build a practical, working ML module that turns citizen-uploaded photos into reliable, auto-verified infrastructure reports for Philippine LGUs.

Last Updated: April 2026
Author: Federex (AyosBayan Research Project)
Status: Active Development – Module 1 (Pothole Detection)

---

## Complete ML Workflow (End-to-End)

This workflow follows the standard machine learning pipeline but is tailored for **real-time citizen reporting systems** and **YOLOv8 object detection**.

### 1. Image Collection (Data Acquisition)
**Objective**: Gather high-quality images of Philippine roads with and without potholes.

**Sources**:
- Public datasets: Roboflow Pothole Detection Dataset v2, RDD2022 (Road Damage Dataset)
- Custom data: Photos collected from AyosBayan citizen reports (anonymized)
- Augmented data: Photos taken in Quezon City / Metro Manila under real conditions (rain, dust, different lighting, angles)

**Target**: Minimum 1,500–2,000 images (70% pothole, 30% non-pothole / normal road)
**Tools**: Roboflow (recommended), Kaggle, Google Drive

**Output**: Raw images folder + `data.yaml` file

---

### 2. Data Processing (Preprocessing)
**Objective**: Clean and prepare images for training.

**Steps**:
- Resize all images to 640×640 (YOLOv8 standard)
- Data augmentation (built-in in YOLOv8):
  - Rotation, flip, brightness/contrast adjustment
  - Noise, blur, rain simulation (important for Philippine weather)
- Split dataset: 70% train / 20% valid / 10% test
- Remove corrupted or irrelevant images (e.g., selfies, cars without road damage)

**Tools**: Ultralytics built-in augmentation + Roboflow preprocessing

**Output**: Clean, augmented dataset ready for training

---

### 3. Feature Engineering
**For YOLOv8 (CNN-based object detection)**:  
**Automatic feature learning** — no manual feature engineering needed.

**What the model automatically learns**:
- Edges and textures of potholes
- Shadows, water inside potholes
- Road surface patterns vs. damage
- Size, shape, and depth cues


---

### 4. Model Selection
**Chosen Model**: **YOLOv8n** (nano) → **YOLOv8s** (small) if more accuracy needed

**Why YOLOv8?**
- Extremely fast inference (< 2 seconds on CPU)
- Excellent accuracy for road damage
- Single-line Python API
- Transfer learning from COCO dataset (already knows roads/objects)
- Built-in support for real-time web deployment

**Alternatives considered**:
- Faster R-CNN (too slow)
- Detectron2 (more complex)
- Custom CNN classifier (no bounding box → less useful for LGUs)

**Final Decision**: YOLOv8n (fast + accurate enough for v1)

---

### 5. Training
**Process**:
1. Load pre-trained YOLOv8n.pt (transfer learning)
2. Train on custom pothole dataset
3. Use Google Colab (free GPU) or local GPU

**Key Training Parameters**:
- Epochs: 50–100
- Image size: 640
- Batch size: 16
- Learning rate: auto (YOLOv8 optimizer)

**Command** (in Colab or terminal):
- bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 name=ayosbayan_pothole


Output: best.pt model file

---

## 6. Evaluation
Metrics (standard for object detection):



MetricTargetMeaning for AyosBayanPrecision> 0.85Low false positives (fewer fake reports flagged wrongly)Recall> 0.85Catch most real potholesmAP@50> 0.90Overall detection qualityF1-Score> 0.85Balance of precision & recallInference Speed< 2 secReal-time citizen upload experience
Confusion Matrix analysis for fake report detection.
Test on Philippine-specific images (rainy roads, night photos, etc.)

---

## 7. Hyperparameter Tuning & Fine-tuning
Methods:

Change model size (yolov8n → yolov8s)
Adjust confidence threshold (default 0.5 → test 0.4–0.7)
More epochs or early stopping
Add more Philippine-specific images if performance drops

Fake Report Logic Tuning:
PythonCONFIDENCE_THRESHOLD = 0.50
if confidence < CONFIDENCE_THRESHOLD or no_detection:
    flag_as_potential_fake = True

---

## 8.  Model Versioning

- v1.0 → pothole detection (YOLOv8n)
- v1.1 → improved dataset
- v2.0 → multi-class detection

Model file naming:
best_v1.pt, best_v2.pt

---

## 9. Deployment & Real-Time Inference (Production Stage)
Integration with AyosBayan:

FastAPI endpoint (/detect-pothole)
Citizen uploads photo → API returns JSON:JSON{
  "detected": true,
  "class": "pothole",
  "confidence": 0.94,
  "is_potential_fake": false,
  "auto_tag": "pothole"
}
LGU dashboard shows bounding box + confidence badge

Deployment Options:

Local / Server (VSCode + Uvicorn)
Docker (future)
Cloud (Render / Railway / AWS)



---

## 10. Monitoring & Continuous Improvement

Log all inference results in database
Collect new citizen photos (with permission) to retrain model every 3–6 months
Track real-world metrics:
% of reports auto-verified
Average verification time saved
Fake report detection rate



