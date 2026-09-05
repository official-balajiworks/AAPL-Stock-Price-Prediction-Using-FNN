
# AAPL Stock Price Prediction Using FNN

A deep learning project that predicts **Apple Inc. (AAPL) next-day closing prices** using historical market data and a Feedforward Neural Network (FNN).

## Project Overview

Stock market prices are influenced by many factors and are difficult to predict accurately. This project explores whether historical OHLCV data can be used to predict the next trading day's closing price.

The model uses a **Feedforward Neural Network** trained on historical AAPL market data.

> **Note:** This project predicts the next trading day's closing price. It is not a financial advice system or a guaranteed stock-price forecasting tool.

## Objectives

- Collect historical AAPL stock data.
- Perform exploratory data analysis.
- Preprocess and scale the features.
- Create a next-day closing-price target.
- Train a Feedforward Neural Network.
- Evaluate the model using regression metrics.
- Visualize actual vs predicted prices.
- Analyze training and validation loss.

## Dataset

The dataset is obtained using the **yfinance** library.

- **Ticker:** AAPL
- **Company:** Apple Inc.
- **Data source:** Yahoo Finance
- **Date range:** January 1, 2000 – December 31, 2026
- **Frequency:** Daily historical market data

### Features

| Feature | Description |
|---|---|
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Closing price |
| Volume | Number of shares traded |

### Target

**Target_Close = Next trading day's Close**

For example:

```text
Today's Close → Tomorrow's Close
     $200     →      $202
```

## Project Workflow

```text
Historical AAPL Data
        ↓
Data Collection
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Create Next-Day Target
        ↓
Chronological Train-Test Split
        ↓
Feature Scaling
        ↓
FNN Model Training
        ↓
Model Evaluation
        ↓
Actual vs Predicted Visualization
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- yfinance
- Scikit-learn
- TensorFlow / Keras
- Google Colab

## Model Architecture

The project uses a Feedforward Neural Network with the following architecture:

```text
Input Layer
5 Features
    ↓
Dense Layer
128 Neurons + ReLU
    ↓
Dropout
20%
    ↓
Dense Layer
64 Neurons + ReLU
    ↓
Dropout
20%
    ↓
Dense Layer
32 Neurons + ReLU
    ↓
Output Layer
1 Neuron
    ↓
Predicted Next-Day Closing Price
```

### Model Configuration

| Parameter | Value |
|---|---|
| Model | Feedforward Neural Network |
| Input features | 5 |
| Hidden layers | 3 |
| Hidden neurons | 128, 64, 32 |
| Activation | ReLU |
| Dropout | 0.2 |
| Output neurons | 1 |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping patience | 5 |

## Data Preprocessing

### 1. Remove unnecessary columns

The `Dividends` and `Stock Splits` columns are removed because they are not used as features in the current model.

### 2. Create the target

```python
hist_timeline['Target_Close'] = hist_timeline['Close'].shift(-1)
```

This shifts the closing price by one trading day so that each row contains the target for the following day.

### 3. Remove the final row

The final row has no next-day closing price, so it is removed.

### 4. Select features

```python
features = ['Open', 'High', 'Low', 'Close', 'Volume']
```

### 5. Chronological train-test split

```python
train_size = int(len(hist_timeline) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]
```

The first 80% of the data is used for training, and the remaining 20% is used for testing.

### 6. Feature scaling

The features are scaled using `MinMaxScaler`.

```python
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data to avoid using test-set information during preprocessing.

## Model Training

The model is trained using:

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
```

Early stopping is used to restore the best-performing model weights:

```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

## Evaluation Metrics

The model is evaluated using:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

### Metric Formulas

**MAE**

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

**MSE**

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

**RMSE**

\[
RMSE = \sqrt{MSE}
\]

**R² Score**

\[
R^2 = 1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

**MAPE**

\[
MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|
\]

## Results

The current FNN evaluation produced the following results:

| Metric | Result |
|---|---:|
| MAE | **2.9731** |
| MSE | **15.9541** |
| RMSE | **3.9943** |
| R² Score | **0.9937** |
| MAPE | **1.5227%** |

### Interpretation

- **MAE = 2.9731:** The average absolute prediction error is approximately **$2.97**.
- **RMSE = 3.9943:** The model's error is approximately **$3.99** when larger errors are penalized more heavily.
- **R² = 0.9937:** The model explains approximately **99.37% of the variation** in the test-set closing prices.
- **MAPE = 1.5227%:** The average percentage error is approximately **1.52%**.

> These results indicate strong agreement between actual and predicted prices on the test set. However, a high R² does not automatically prove that the model can reliably forecast future stock prices.

## Visualizations

### 1. Actual vs Predicted Closing Price

This graph compares the actual AAPL closing prices with the prices predicted by the FNN.

The predicted line follows the actual price trend closely, including major rises and falls.

### 2. Training vs Validation Loss

This graph shows the training and validation loss across epochs.

It helps identify whether the model is learning effectively and whether overfitting may be occurring.

## Project Structure

```text
AAPL-Stock-Prediction/
│
├── AAPL_Stock_Prediction.ipynb
├── README.md
└── requirements.txt
```

## Installation

Install the required libraries:

```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
```

## How to Run

1. Open the notebook in **Google Colab** or Jupyter Notebook.
2. Install the required libraries.
3. Run the data collection section.
4. Perform exploratory data analysis.
5. Preprocess the dataset.
6. Train the FNN model.
7. Evaluate the model.
8. View the actual vs predicted price graph.

## Important Limitations

- Stock prices are influenced by many external factors.
- Historical OHLCV data alone may not capture news, market sentiment, or economic events.
- The model predicts the next trading day's closing price, not long-term market direction.
- A close visual fit does not guarantee reliable future forecasting.
- The current evaluation should be compared with a simple baseline that predicts tomorrow's close as today's close.
- The reported results should not be interpreted as guaranteed trading performance.

## Future Improvements

- Compare the FNN with a naive baseline.
- Add technical indicators such as RSI, MACD, and moving averages.
- Predict daily returns instead of raw closing prices.
- Experiment with LSTM and GRU models.
- Use longer historical sequences.
- Perform walk-forward validation.
- Compare multiple deep learning architectures.
- Analyze prediction errors during periods of high volatility.
- Add a simple Streamlit interface for predictions.

## Conclusion

This project demonstrates the use of a Feedforward Neural Network for next-day AAPL closing-price prediction. The model achieved strong test-set performance and closely followed the actual price trend.

However, further validation and comparison with a naive baseline are necessary to determine whether the FNN provides meaningful predictive improvement.

## Disclaimer

This project is intended for **educational and research purposes only**. It does not provide financial advice, investment recommendations, or guaranteed stock-price predictions.

## Author

**Your Name**

**GitHub:** [Your GitHub Profile](https://github.com/)

**LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/)
