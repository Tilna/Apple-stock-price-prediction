#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "jax"
from tf_keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# In[11]:


## load dataset
df = pd.read_csv("P625 DATASET.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)


# In[12]:


# Load trained LSTM model
model = load_model("lstm_stock_model.keras", compile=False)


# In[13]:


# App title
st.title("📈 Apple Stock Price Forecasting (LSTM)")
st.write(
    "Select a date range to visualize historical prices and forecast future stock prices "
    f"(Available range: {df.index.min().date()} to {df.index.max().date()}).")


# In[14]:


# Date selection
start_date, end_date = st.date_input(
    "Select Date Range",value=[df.index.min(), df.index.max()],min_value=df.index.min(),max_value=df.index.max())


# In[15]:


# Convert to pandas timestamps
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)


# In[16]:


# Validate date range
if start_date >= end_date:
    st.error("❌ Start date must be earlier than end date.")
    st.stop()


# In[17]:


# Filter data for visualization
filtered_df = df.loc[start_date:end_date]


# In[18]:


# Plot historical prices
st.subheader("📊 Historical Stock Prices")
fig, ax = plt.subplots()
ax.plot(filtered_df.index, filtered_df['Close'], label="Historical Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)


# In[21]:


# Forecast Button

if st.button("🔮 Forecast Next 30 Days"):
    window_size = 60
    # Use data only up to selected end date
    selected_df = df.loc[:end_date]
    if len(selected_df) < window_size:
        st.error("❌ Not enough historical data before the selected end date to generate a forecast.")
        st.stop()

    # Scale selected data
    scaler = MinMaxScaler()
    scaled_selected = scaler.fit_transform(selected_df[['Close']])

    # Take last window from selected range
    last_window = scaled_selected[-window_size:]
    current_input = last_window.reshape(1, window_size, 1)

    # Recursive forecasting
    future_predictions = []

    for _ in range(30):
        next_pred = model.predict(current_input, verbose=0)[0][0]
        future_predictions.append(next_pred)

        current_input = np.append(
            current_input[:, 1:, :],
            [[[next_pred]]],
            axis=1)

    # Inverse scaling
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1))

    # Generate future dates AFTER selected end date
    future_dates = pd.date_range(
        start=end_date,
        periods=31,freq='B')[1:]

    forecast_df = pd.DataFrame(
        future_predictions,
        index=future_dates,
        columns=["Forecast"])

    # Forecast Plot
    st.subheader("📈 30-Day Stock Price Forecast")
    fig2, ax2 = plt.subplots()
    ax2.plot(
    filtered_df.index,filtered_df['Close'],label="Historical (Selected Range)",color="blue")
    ax2.plot(forecast_df.index, forecast_df['Forecast'], label="Forecast", color="red")
    ax2.axvline(end_date, color='gray', linestyle='--', label="Forecast Start")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)
    
    # Forecast Table
    st.subheader("📋 Forecasted Values")
    st.dataframe(forecast_df)


# In[ ]:




