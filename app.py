import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model(r'C:\Users\Admin\OneDrive\Desktop\stock_market_analysis\Stock_Predictions_Model_gru.keras')

# Streamlit App Interface
st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2014-05-29'
end = '2024-05-30'  

# Download Stock Data
data = yf.download(stock, start, end)

# Display Stock Data
st.subheader('Stock Data')
st.write(data)

# Prepare Data for Prediction
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)
data_test_scaled = scaler.transform(data_test)


# Here we create the graph of MA50
st.subheader('Price vs MA50')
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data.Close, 'g', label='Price')
ax1.plot(data.Close.rolling(50).mean(), 'r', label='MA50')
ax1.set_title('Price vs MA50')
ax1.legend()

# Here we create the graph of MA100
fig2, ax2 = plt.subplots(figsize=(8, 6)) 
ax2.plot(data.Close, 'g', label='Price')
ax2.plot(data.Close.rolling(100).mean(), 'r', label='MA100')
ax2.set_title('Price vs MA100')
ax2.legend()

# Here we create the graph of MA200
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(data.Close, 'g', label='Price')
ax3.plot(data.Close.rolling(200).mean(), 'r', label='MA200')
ax3.set_title('Price vs MA200')
ax3.legend()

# Now you can display the figures
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)

# Prepare test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Prepare x_test and y_test
x_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i - 100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_pred_scaled = model.predict(x_test)

# Reverse scaling
scale = 1 / scaler.scale_[0]
y_pred = y_pred_scaled * scale
y_test = y_test * scale

# Function to generate prediction for the latest data
def generate_prediction(model, x):
    """Generates prediction using the provided model."""
    predict = model.predict(x)
    return predict[0]

predicted_price = generate_prediction(model, x_test[-1].reshape(1, x_test.shape[1], x_test.shape[2]))

# Display Prediction Results
st.subheader("Stock Prediction")
predicted_price_actual = predicted_price[0] * scale
predicted_price_actual /= 10
st.write(f"The model predicts the stock price might be around {predicted_price_actual:.3f} USD")

# Plot Original vs Predicted Price
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test, label='Original Price')
ax.plot(y_pred, label='Predicted Price')
ax.set_title('Original vs Predicted Price')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Display predicted price
st.write(f"Predicted Price: {predicted_price_actual:.3f}")

# Stock Market Information
st.write("This is the sample of Stock Price Prediction and Analysis. \n")
st.write("**About the Stock Market:**")
st.write(
    """The stock market is a dynamic and complex environment where the prices of publicly traded companies constantly fluctuate based on a multitude of factors. While past performance and financial indicators can provide valuable insights, predicting stock prices with absolute certainty is impossible. This Streamlit application utilizes a machine learning model to generate predictions about future stock prices based on historical data. It's important to understand that these predictions are estimates and should be used in conjunction with thorough research and professional guidance before making any investment decisions."""
)
