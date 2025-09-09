# Day 1: Neural Networks Workshop - Breast Cancer Classification

---

## Overview

This project implements a **feedforward neural network** to perform binary classification on the Breast Cancer dataset available from scikit-learn. The objective is to build, train, and evaluate a neural network that can accurately classify tumors as malignant or benign.

The notebook covers data loading, preprocessing, model design and training, and detailed evaluation using multiple performance metrics and visualization techniques.

---

## Dataset

The **Breast Cancer Wisconsin (Diagnostic)** dataset contains 569 samples with 30 numeric features capturing characteristics of cell nuclei present in digitized images of breast masses. The classification task is binary, distinguishing between malignant (cancerous) and benign (non-cancerous) tumors.

This dataset is included in scikit-learn and does not require manual download.

---

## Components and Methodology

### 1. Data Loading and Preprocessing

- **Loading Dataset:** Used `load_breast_cancer()` from scikit-learn to load features and labels.
- **Feature Scaling:** Applied `StandardScaler` to standardize features with zero mean and unit variance, enabling stable and efficient training.
- **Train-Test Split:** Divided data into training (80%) and testing (20%) sets using `train_test_split` for unbiased model evaluation on unseen data.

### 2. Neural Network Architecture

- Built the model using TensorFlow Keras Sequential API.
- Architecture details:
  - Input layer sized to dataset features.
  - Two hidden Dense layers (32 and 16 neurons) with **ReLU** activation to capture non-linear feature interactions.
  - Output layer with a single neuron and **Sigmoid** activation to output binary classification probabilities.

### 3. Model Compilation and Training

- Used the **Adam** optimizer, known for adaptive learning rates and momentum, improving convergence speed and stability.
- The loss function was **binary cross-entropy**, suitable for gauging error in binary classification tasks.
- Model trained for 30 epochs with a batch size of 16 and 10% validation split to monitor generalization during training.

### 4. Model Evaluation

- Predicted labels on the test set.
- Computed:
  - **Accuracy:** Overall correctness of classification.
  - **Precision:** Ratio of true positives among predicted positives.
  - **Recall:** Proportion of actual positives correctly identified.
  - **F1-Score:** Harmonic mean of precision and recall, balancing the trade-off.
- Visualized performance using a **Confusion Matrix** showing counts of true/false positives and negatives.

---

## Core Concepts Covered

- **Feature Scaling**: Standardizing inputs helps neural networks learn more effectively.
- **Activation Functions**: ReLU mitigates vanishing gradient problems; Sigmoid outputs probabilities.
- **Binary Cross-Entropy Loss**: Tailored for binary output probability comparison.
- **Adam Optimizer**: Combines adaptive learning rates and momentum for better training efficiency.
- **Train-Test Split**: Essential for unbiased evaluation.
- **Confusion Matrix & Metrics**: Provide detailed insight beyond accuracy, especially for imbalanced data.

---

## How to Use This Repository

1. Open the Jupyter Notebook inside `day1-neural-networks-workshop/`.
2. Run all cells sequentially to recreate the analysis.
3. Read the Markdown cells for in-depth explanations and rationale.
4. View output metrics and confusion matrix for assessment.

---

## Dependencies

- Python 3.x
- scikit-learn
- TensorFlow (Keras)
- NumPy
- Matplotlib
- Pandas

Install dependencies using: **pip install numpy pandas matplotlib scikit-learn tensorflow**

