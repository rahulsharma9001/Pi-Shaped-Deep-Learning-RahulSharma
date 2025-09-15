# Machine Learning Model Pruning and Confusion Matrix Visualization

## Project Overview
This project demonstrates pruning a machine learning model to optimize its size and performance, followed by evaluating the pruned model using a confusion matrix visualization. The key objectives include:

- Using a pruned model for making predictions on test data.
- Computing and plotting the confusion matrix with a blue color map for clear visualization.
- Saving the plot with a meaningful filename following project naming conventions.

---

## Technologies Used
- Python 3.7+
- TensorFlow/Keras (for model pruning and inference)
- NumPy
- scikit-learn
- Matplotlib

---

## Installation & Setup

1. Clone the repository:
```bash
    https://github.com/rahulsharma9001/Pi-Shaped-Deep-Learning-RahulSharma.git
```

```bash
    cd Pi-Shaped-deep-learning/Day-4/
```


2. Install dependencies:

```bash
    pip install numpy scikit-learn matplotlib tensorflow
```


3. Ensure your pruned model `model_pruned` and test data `x_test`, `y_test` are available.

---

## Usage Example

```bash
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
```

### Predict on test set using pruned model
```bash
    y_pred = np.argmax(model_pruned.predict(x_test), axis=1)
```
### Compute confusion matrix
```bash
    cm = confusion_matrix(y_test, y_pred)
```

### Plot confusion matrix with blue colormap
```bash
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('projectname_confusion_matrix_blues_2025-09-15.png')
    plt.show()
```

---

## Core Concept Questions and Answers

### 1. Why is hyperparameter tuning important, and what trade-offs does it involve?
Hyperparameter tuning optimizes model performance by adjusting parameters controlling learning behavior (e.g., learning rate, batch size). The trade-off lies between extensive search time/computational resources and improved accuracy/generalization. Over-tuning also risks overfitting to validation data.

### 2. How does model pruning or compression impact performance and resource usage?
Pruning reduces model size by removing redundant or less important weights and neurons. It results in lower memory usage and faster inference, which is crucial for deployment on resource-constrained devices. However, excessive pruning may degrade accuracy if important components are removed. Careful balancing and retraining help maintain performance.

### 3. Why is dropout effective in preventing overfitting?
Dropout randomly disables neurons during training, forcing the network to develop redundant representations and reducing co-adaptation among neurons. This regularization technique helps generalization and prevents the model from fitting too closely to training data noise.

### 4. What challenges arise when deploying deep learning models in production?
Challenges include latency requirements, memory and compute constraints, model robustness to varying data, maintainability, updating models without downtime, and integration with existing systems.

### 5. How does TensorFlow Lite (or ONNX, TorchScript) help in deployment optimization?
These frameworks convert models into optimized, platform-specific formats for efficient inference on edge devices or mobile platforms. They reduce model size, enhance execution speed, and enable deployment on hardware lacking full Python environments.

### 6. What is the balance between model accuracy and efficiency in real-world applications?
A high-accuracy model may be computationally expensive and slow, unsuitable for real-time or low-power devices. Efficiency involves compromising some accuracy to meet latency, power, or memory constraints, ensuring practical usability.

### 7. How can hardware (GPU, TPU, Edge devices) influence optimization strategies?
Hardware capabilities dictate optimization choices: GPUs excel at parallel floating-point operations, TPUs specialize in tensor processing for deep learning, while edge devices require lightweight, pruned, or quantized models for limited resources.

### 8. Looking ahead, how might optimization differ for Transformer-based models compared to CNNs/RNNs?
Transformer models are typically larger and require different pruning/quantization techniques focusing on attention mechanisms and large parameter matrices. Optimization might involve sparsity in attention heads and layers, knowledge distillation, and novel architectures to reduce complexity.

---

## Results and Evaluation
The confusion matrix plot reveals the classification performance of the pruned model, highlighting true and false predictions across classes with a clear blue gradient for visibility.
