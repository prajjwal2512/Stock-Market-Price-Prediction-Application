import numpy as np
import pandas as pd
from keras.models import load_model 
import matplotlib.dates as mdates
import streamlit as st 
import seaborn as sns

# Load the model
model = load_model(r'C:\Users\Lenovo Laptop\Desktop\SP\Stock Prediction Model.keras')

# Set the page title and icon
st.set_page_config(page_title='Stock Market Predictor', page_icon='📈', layout='centered')

# CSS to inject contained in a string
# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align: center; color: #00A5FF;">📉Stock Market Predictor📈</h1>', unsafe_allow_html=True)

# User input for stock symbol
st.markdown('<h2 style="text-align: center; color: #4CAF50;">Enter Stock Symbol👇</h2>', unsafe_allow_html=True)
stock = st.text_input('', 'TSLA')


# Date range
start = '2012-01-01'
end = '2024-06-10'

# Fetch data
data = yf.download(stock, start, end)

# Display data
st.subheader('Stock data')
st.write(data)

# Data preparation
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pass_100_days = data_train.tail(100)
data_test = pd.concat([pass_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Plot Price vs MA50
st.subheader('Price vs MA50')

fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'red', label='50 Days MA')
plt.plot(data.Close, 'green', label='Close Price')
plt.xlabel('Days (2012-2024)')
plt.ylabel('Price ($)')
plt.legend(loc='upper left')
plt.xticks(pd.date_range(start='2012-01-01', end='2024-01-01', freq='2YS').to_pydatetime(), rotation=45)
plt.show()
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'red', label='50 Days MA')
plt.plot(ma_100_days, 'black', label='100 Days MA')
plt.plot(data.Close, 'green', label='Close Price')
plt.xlabel('Days (2012-2024)')
plt.ylabel('Price ($)')
plt.legend(loc='upper left')
plt.xticks(pd.date_range(start='2012-01-01', end='2024-01-01', freq='2YS').to_pydatetime(), rotation=45)
plt.show()
st.pyplot(fig2)

# Plot Price vs MA50 vs MA100 vs MA200
st.subheader('Price vs MA50 vs MA100 vs MA200')
fig3 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'red', label='50 Days MA')
plt.plot(ma_100_days, 'blue', label='100 Days MA')
plt.plot(ma_200_days, 'black', label='200 Days MA')
plt.plot(data.Close, 'green', label='Close Price')
plt.xlabel('Days (2012-2024)')
plt.ylabel('Price ($)')
plt.legend(loc='upper left')
plt.xticks(pd.date_range(start='2012-01-01', end='2024-01-01', freq='2YS').to_pydatetime(), rotation=45)
plt.show()
st.pyplot(fig3)

# Preparing data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Prediction
predict = model.predict(x)

# Scaling back to original
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10, 8))
plt.plot(data.index[-len(predict):], predict, 'green', label='Predicted Price')
plt.plot(data.index[-len(y):], y, 'red', label='Original Price')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend(loc='upper center')
#plt.xticks(pd.date_range(start='2012-01-01', end='2024-01-01', freq='2YS').to_pydatetime(), rotation=45)
plt.show()
st.pyplot(fig4)


