# Day 2: Recurrent Neural Networks (RNN) Workshop - Tesla Stock Price Prediction

## Overview

This assignment focuses on building and training a Recurrent Neural Network (RNN), specifically an LSTM model, to predict Tesla's next-day stock opening prices. Using historical stock price data, the model learns temporal patterns, enabling time-series forecasting, a key application of RNNs.

---

## Dataset

The dataset contains Tesla stock prices with daily records including open, high, low, close prices, and volume. The data is sourced from Kaggle and includes date information which is parsed and set as the index.

---

## Step-by-Step Methodology

### 1. Import Libraries

We import essential libraries:
- `pandas` and `numpy` for data manipulation.
- `MinMaxScaler` for scaling prices to [0,1] to stabilize training.
- Keras `Sequential`, `LSTM`, and `Dense` for building the RNN.
- `matplotlib` and sklearn metrics for visualization and evaluation.

### 2. Load and Inspect Data

The dataset CSV is loaded into a pandas DataFrame, with the 'Date' column parsed as datetime and set as the index. We print the first few rows to understand the data structure.

### 3. Data Preprocessing

- Extract the 'Open' price column as the prediction target.
- Scale prices to a normalized range using MinMaxScaler, which helps the neural network converge more efficiently.
- Create sequences where each input is 60 days of consecutive prices, and the label is the 61st day’s price, framing the problem as supervised learning.
- Split data into training and testing sets (80%-20%) while preserving temporal order—shuffling is avoided because time sequence matters.
- Reshape input to the 3-dimensional format expected by LSTM layers: (samples, time steps, features).

### 4. Build the LSTM Model

- The model has two stacked LSTM layers (50 units each) to capture complex temporal dependencies.
- The last layer is a Dense layer with one neuron for predicting the stock price.
- The model uses the Adam optimizer and mean squared error loss, suitable for regression problems.

### 5. Train the Model

The model is trained for 30 epochs with batch size 32 and a 10% validation split. Early stopping or dropout could be added to avoid overfitting.

### 6. Evaluate the Model

- Predictions are generated on the test set.
- Both predictions and true values are inverse-transformed to the original price scale.
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) quantify prediction accuracy.

### 7. Visualize Predictions

Plotting predicted versus actual opening prices provides visual insight into the model's performance over time to analyze prediction trends and deviations.

---

## Core Concepts Explained

- **Why LSTM?**  
  LSTMs handle sequence data and long-term dependencies better than traditional feedforward networks, making them suitable for stock price forecasting.

- **Feature Scaling Impact:**  
  Scaling input data prevents large values from skewing gradients and accelerates convergence.

- **Sequence Framing:**  
  Breaking the time series into input-output sequences helps the model learn from temporal context.

- **Overfitting Assessment:**  
  Monitoring validation loss during training and visualizing predictions help detect overfitting.

- **Regression Metrics:**  
  MAE and RMSE provide interpretable, quantitative measures of errors in price predictions.

---

## Usage Instructions

1. Place the dataset CSV (`Tesla.csv`) which is a Stock Price CSV file kept in the same folder as the notebook or provide the correct file path.
2. Run the notebook cells sequentially to preprocess data, build, train, evaluate, and visualize the LSTM model.
3. Review markdown explanations for understanding each step.

---

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- TensorFlow (Keras)

Install them with: ** pip install pandas numpy scikit-learn matplotlib tensorflow**
