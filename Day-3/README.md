# Day 3: Convolutional Neural Networks (CNN) Workshop - Fashion-MNIST Classification

## Overview

This assignment demonstrates building and training a CNN for image classification on the Fashion-MNIST dataset. The dataset contains 70,000 grayscale images (28x28 pixels) representing 10 different fashion categories (e.g., T-shirt, sneaker, bag).

The goal is to classify images correctly into these 10 categories using a CNN and evaluate the model’s performance with classification metrics including accuracy and confusion matrix.

---

## Step-by-Step Workflow

### 1. Importing Libraries and Loading Data

We import TensorFlow/Keras for model building and training, Matplotlib for visualization, and sklearn for evaluation.  
The Fashion-MNIST dataset is loaded easily from Keras datasets with predefined train-test splits.

### 2. Data Preprocessing

- Normalize pixel values from [0, 255] to [0,1] for faster convergence and better training stability.
- Reshape dataset images from (28,28) to (28,28,1) to include the single-channel dimension expected by CNN layers.
- Convert categorical labels into one-hot encoded vectors for multi-class classification.

### 3. CNN Model Architecture

- The model has two convolutional layers with ReLU activations and MaxPooling layers to extract spatial features and reduce dimensionality.
- The Flatten layer converts 2D feature maps to 1D feature vectors for Dense layers.
- Dense layers including a dropout layer reduce overfitting.
- The output layer uses softmax activation for multi-class probability output across 10 classes.

### 4. Model Compilation and Training

- The model uses the Adam optimizer and categorical cross-entropy loss function suitable for multi-class classification.
- Trained for 10 epochs with a batch size of 64 using 10% of training data for validation.

### 5. Model Evaluation

- Performance is measured on test data by accuracy.
- Confusion matrix visualizes correct and incorrect predictions across classes, helping analyze class-wise performance.

### 6. Data Augmentation (Optional)

- Using Keras’ ImageDataGenerator, data augmentation artificially expands the training set by applying random transformations (roll, zoom, shifts, flips).
- Augmentation improves generalization by reducing overfitting and makes the model robust to variations in input images.

---

## Core Concepts Addressed

- CNNs effectively capture spatial hierarchies in images via convolutional and pooling layers.
- Convolutional filters learn spatial features like edges and patterns.
- Pooling downsamples feature maps, reducing parameters and computational load while maintaining important information.
- Normalizing pixel data leads to more stable and faster model convergence.
- Softmax outputs probabilities for each class in multi-class settings.
- Data augmentation acts as regularization, helping prevent overfitting.
- The confusion matrix provides detailed insight into class-wise classification performance beyond overall accuracy.

---

## How to Run This Project

1. Clone or download the repository.
2. Ensure dependencies (`tensorflow`, `matplotlib`, `scikit-learn`) are installed.
3. Run the Jupyter Notebook `Fashion-MNIST CNN assignment.ipynb` cells sequentially.
4. Follow Markdown explanations for clarity.
5. Compare model performance with and without data augmentation.

---

## Dependencies

- Python 3.x  
- TensorFlow 2.x (with Keras)  
- scikit-learn  
- matplotlib

Install dependencies with:  **pip install tensorflow scikit-learn matplotlib**
