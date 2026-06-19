# 📈 Apple Stock Price Forecasting

A deep learning project that predicts Apple (AAPL) stock prices using an LSTM (Long Short-Term Memory) neural network, with an interactive Streamlit web application for visualization and forecasting.

---

## 🔍 Project Overview

This project uses historical Apple stock market data (2012–2024) to train an LSTM model capable of predicting future closing prices. The trained model is deployed via a Streamlit web app that allows users to select a date range, visualize historical prices, and generate a 30-day stock price forecast.

---

## 🚀 Features

- 📊 Interactive date range selection for historical price visualization
- 🤖 LSTM-based deep learning model for time series forecasting
- 🔮 Recursive 30-day future price prediction
- 📉 Side-by-side plot of historical vs. forecasted prices
- 📋 Forecast table with predicted values for all 30 days

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Deep Learning | TensorFlow, Keras (LSTM) |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn (MinMaxScaler) |
| Visualization | Matplotlib |
| Web App | Streamlit |

---

## 📁 Project Structure

```
Apple-stock-price-prediction/
│
├── Apple_stock_price_forecast_app.py   # Streamlit web application
├── lstm_stock_model.keras              # Trained LSTM model
├── P625_DATASET.csv                    # Historical AAPL stock data (2012–2024)
└── README.md
```

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tilna/Apple-stock-price-prediction.git
cd Apple-stock-price-prediction
```

### 2. Install dependencies

```bash
pip install streamlit tensorflow scikit-learn pandas numpy matplotlib
```

### 3. Run the app

```bash
streamlit run Apple_stock_price_forecast_app.py
```

---

## 🧠 Model Architecture

- **Model Type:** LSTM (Long Short-Term Memory)
- **Input:** 60-day rolling window of closing prices
- **Output:** Next day's predicted closing price
- **Forecasting:** Recursive prediction for 30 future business days
- **Scaling:** MinMaxScaler (0–1 normalization)

---

## 📊 Dataset

- **Source:** Apple Inc. (AAPL) historical stock data
- **Period:** January 2012 – 2024
- **Records:** ~2,010 trading days
- **Features used:** `Close` price

---

## 📌 How It Works

1. User selects a date range in the app
2. The app filters historical data and scales it using MinMaxScaler
3. The last 60 days of the selected range are fed as input to the LSTM model
4. The model recursively predicts the next 30 business days
5. Predictions are inverse-transformed back to actual price values and plotted

---

## 🙋 Author

**Tilna**
- GitHub: [@Tilna](https://github.com/Tilna)

---

