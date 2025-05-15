# Alzheimer’s Disease Detection and Analysis Tool

This project implements a deep learning-based application for detecting Alzheimer’s Disease (AD) from brain MRI scans. Using convolutional neural networks (CNNs) for image classification and analytical visualization, the tool assists researchers and clinicians in early-stage detection and data-driven analysis of Alzheimer’s progression.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Dataset Preparation](#dataset-preparation)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Methods Overview](#methods-overview)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Features

* **MRI-based Classification:** Trains a CNN (e.g., ResNet-50) to classify MRI scans into categories: **Control**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer’s Disease (AD)**.
* **Data Augmentation:** Applies transformations (flips, rotations, intensity shifts) to improve model robustness on limited medical data.
* **Visualization Dashboard:** Generates performance plots (accuracy/loss curves), confusion matrix, and Grad-CAM heatmaps highlighting regions of interest.
* **Modular Pipeline:** Separate modules for data loading, preprocessing, model definition, training, and evaluation.

---

## Prerequisites

* Python 3.8 or higher
* PyTorch or TensorFlow (choose framework in config)
* OpenCV
* NumPy, Pandas, Matplotlib, Seaborn
* scikit-learn

---

## Installation

1. **Clone Repository**

   ```bash
   git clone https://github.com/yourusername/alzheimer-detection-tool.git
   cd alzheimer-detection-tool
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

1. **Acquire MRI Data**

   * Download the OASIS or ADNI dataset following their data use agreements.

2. **Organize Directory Structure**

   ```text
   data/
   ├── train/
   │   ├── Control/
   │   ├── MCI/
   │   └── AD/
   └── test/
       ├── Control/
       ├── MCI/
       └── AD/
   ```

3. **Preprocessing**

   * Resize images to 224×224
   * Normalize pixel intensities
   * Save preprocessed arrays or HDF5 files for faster loading.

---

## Model Training

1. **Configuration**

   * Modify `config.yaml` to set framework, hyperparameters (batch size, learning rate, epochs).

2. **Run Training Script**

   ```bash
   python train.py --config config.yaml
   ```

3. **Checkpointing**

   * Models saved to `checkpoints/` as `model_epoch_{n}.pth`.

---

## Evaluation

* **Test Metrics:** Generates accuracy, precision, recall, and F1-score on test set.
* **Confusion Matrix:** Saved to `plots/confusion_matrix.png`.
* **Grad-CAM:** Creates heatmaps showing salient regions for AD predictions.

To run evaluation only:

```bash
python evaluate.py --checkpoint checkpoints/model_epoch_10.pth
```

---

## Usage

After training or loading a pretrained model, you can classify individual MRI images:

```bash
python predict.py --model checkpoints/model_latest.pth --image sample_mri.jpg
```

Output:

* Predicted class label and confidence score.
* Optional: Display Grad-CAM overlay on input image.

---

## Project Structure

```
├── data/                 # MRI dataset folders
├── checkpoints/          # Saved model weights
├── plots/                # Training/evaluation visualizations
├── config.yaml           # Hyperparameter and paths config
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── predict.py            # Single-image inference
├── models.py             # CNN definitions (ResNet variants)
└── utils.py              # Data loaders, transforms, Grad-CAM
```

---

## Methods Overview

* **`models.py`**: Defines CNN architectures (ResNet-18, ResNet-50) with final classification head.
* **`utils.py`**:

  * `load_data()` – PyTorch/TensorFlow data pipelines with augmentation.
  * `compute_metrics()` – Calculates performance metrics.
  * `generate_gradcam()` – Produces heatmap explanations.
* **`train.py`**:

  * Initializes model, optimizer, scheduler.
  * Loops over epochs: forward, backward, logging, checkpointing.
* **`evaluate.py`**:

  * Loads checkpoint, runs inference on test set, saves plots.
* **`predict.py`**:

  * Single-image prediction with optional Grad-CAM display.

---

## Troubleshooting

* **CUDA Errors**:

  * Verify GPU availability and correct CUDA toolkit.
  * If unavailable, set `device: cpu` in `config.yaml`.
* **Data Loading Issues**:

  * Ensure directory names match labels in `config.yaml`.
  * Check image formats (e.g., `.jpg`, `.png`).
* **Model Divergence**:

  * Adjust learning rate or switch optimizer (Adam, SGD).
  * Increase data augmentation to reduce overfitting.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
