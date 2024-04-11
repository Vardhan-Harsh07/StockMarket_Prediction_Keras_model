import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime
from keras.models import load_model
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Data Collection and Preprocessing

yf.pdr_override()

st.title("Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker", 'AAPL')

start = st.date_input("Start", value = pd.to_datetime('2021-01-01'))
end = st.date_input("End", value = pd.to_datetime('today'))

df = pdr.get_data_yahoo(user_input, start, end)

st.title("Stock Market Data Summary")
summary_statistics = pdr.describe()
st.write(summary_statistics)
