import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# Descriptive Statistics  (mean, median, standard deviation, etc.) 
def Descriptive_Stats(df, data):
    
    summary_statistics = df.describe()
    
    st.title("Stock Market Data Summary")

    # Display summary statistics table
    st.write("Summary Statistics:")
    st.write(summary_statistics)

    # Create bar plots for mean and median>
    st.write("Mean and Median Values:")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    mean_values = data.mean()
    median_values = data.median()

    mean_values.plot(kind='bar', ax=ax[0])
    ax[0].set_title("Mean Values")
    ax[0].set_xlabel("Columns")
    ax[0].set_ylabel("Mean")
    ax[0].grid(axis='y')

    median_values.plot(kind='bar', ax=ax[1])
    ax[1].set_title("Median Values")
    ax[1].set_xlabel("Columns")
    ax[1].set_ylabel("Median")
    ax[1].grid(axis='y')

    st.pyplot(fig)

# Probability Distributions (e.g., normal distribution, log-normal distribution, outliers, z-scores).
def Prob_Distributions(df, data):
    stock_returns = data
    
    st.title("Stock Returns Distribution Analysis")

    # Display the distribution of stock returns
    st.write("Distribution of Stock Returns:")
    fig = plt.figure(figsize = (12,6))
    plt.hist(stock_returns, bins=30, edgecolor='k')
    st.pyplot(fig)

    # Fit the data to probability distributions
    st.write("Fit to Probability Distributions:")

    # Fit to a normal distribution
    loc, scale = stats.norm.fit(stock_returns)
    normal_dist = stats.norm(loc=loc, scale=scale)

    # Fit to a log-normal distribution
    shape, loc, scale = stats.lognorm.fit(stock_returns, floc = 0)
    lognormal_dist = stats.lognorm(s=shape, loc=loc, scale=scale)

    # Create a histogram for the data
    hist_values, bin_edges = np.histogram(stock_returns, bins=30, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Plot the histogram and the fitted distributions
    fig2 = plt.figure(figsize=(8, 6))
    plt.hist(stock_returns, bins=30, density=True, alpha=0.6, label='Histogram')
    x = np.linspace(np.min(stock_returns), np.max(stock_returns), 100)
    plt.plot(x, normal_dist.pdf(x), 'r-', label='Fitted Normal Distribution')
    plt.plot(x, lognormal_dist.pdf(x), 'g-', label='Fitted Log-Normal Distribution')
    plt.xlabel("Returns")
    plt.ylabel("Probability Density")
    plt.legend()
    st.pyplot(fig2)
    
    # Outlier detection using Z-scores
    st.write("Outlier Detection for Closing values:")
    z_scores = (stock_returns - np.mean(stock_returns)) / np.std(stock_returns)
    outliers = (np.abs(z_scores) > 2)  # Adjust the threshold as needed

    # Display the outliers
    outlier_indices = np.where(outliers)[0]

    # Plot outliers in a scatter plot
    fig3 = plt.figure(figsize=(8, 6))
    plt.scatter(range(len(stock_returns)), stock_returns['Close'], c='b', label='Data')
    plt.scatter(outlier_indices, stock_returns['Close'][outlier_indices], c='r', label='Outliers', marker='x')
    plt.xlabel("Data Point Index (Closing Values)")
    plt.ylabel("Returns")
    plt.legend()
    st.pyplot(fig3)
    
def Hypothesis_Test(df, data):
    
    # Sample dataset of stock returns
    stock_returns = data  # Example data, replace with your actual data

    # Streamlit app
    st.title("Hypothesis Testing for Stock Market Anomalies")

    # Null hypothesis: Stock returns follow a normal distribution (Efficient Market Hypothesis or EMH)
    # Alternative hypothesis: Stock returns do not follow a normal distribution (indicating a potential anomaly)

    # Perform a normality test (e.g., Shapiro-Wilk test)
    statistic, p_value = stats.shapiro(stock_returns)

    # Set the significance level (alpha)
    alpha = 0.05

    # Display the results
    st.write("Shapiro-Wilk Normality Test Results:")
    st.write(f"Test Statistic: {statistic}")
    st.write(f"P-Value: {p_value}")

    if p_value < alpha:
        st.write("Reject the null hypothesis: Stock returns do not follow a normal distribution.")
        st.write("This suggests a potential market anomaly.")
    else:
        st.write("Fail to reject the null hypothesis: Stock returns may follow a normal distribution.")
        st.write("This is in line with the Efficient Market Hypothesis (EMH).")

    # Plot the stock returns
    fig = plt.figure(figsize=(8, 6))
    plt.hist(stock_returns, bins=30, edgecolor='k')
    plt.xlabel("Stock Returns")
    plt.ylabel("Frequency")
    plt.title("Histogram of Stock Returns")
    st.pyplot(fig)
    
def plot_raw_data(df):    # Add this
	fig1 = go.Figure()
	fig1.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
	fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
	fig1.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig1)
 
def plot_rateofreturn(df):    # Add this
    closing_price = df['Close']
    returns = np.log(closing_price).diff()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=returns, name="rate of returns"))
    fig2.layout.update(title_text='Rate of Returns', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

def plot_moving_averages(df):
    ma100 = df.Close.rolling(100).mean()   # 100 days moving average
    ma200 = df.Close.rolling(200).mean()   # 200 days moving average
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y = df['Close']))
    fig2.add_trace(go.Scatter(x=df['Date'], y = ma100, name="100 days moving average"))
    fig2.add_trace(go.Scatter(x=df['Date'], y = ma200, name="200 days moving average"))
    fig2.layout.update(title_text='Rate of Returns', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

    
# def training_data(df):        
    
#     ''' This function was used to create the model and train it on training data. The model has already been built and saved to avoid going through the same steps repeatedly. '''
    
#     data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
#     data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])
#     scaler = MinMaxScaler(feature_range=(0,1))
#     data_training_array = scaler.fit_transform(data_training)
    
#     x_train = []
#     y_train = []

#     for i in range(100, data_training_array.shape[0]):
#         x_train.append(data_training_array[i-100 : i])
#         y_train.append(data_training_array[i , 0])
        
#     x_train, y_train = np.array(x_train), np.array(y_train)
    
#     model = Sequential()
#     model.add(LSTM(units = 50, activation = "relu", return_sequences = True, 
#                 input_shape = (x_train.shape[1], 1)))
#     model.add(Dropout(0.2))

#     model.add(LSTM(units = 60, activation = "relu", return_sequences = True))
#     model.add(Dropout(0.3))

#     model.add(LSTM(units = 80, activation = "relu", return_sequences = True))
#     model.add(Dropout(0.4))

#     model.add(LSTM(units = 120, activation = "relu"))
#     model.add(Dropout(0.5))

#     model.add(Dense(units = 1))
    
#     model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#     model.fit(x_train, y_train, epochs = 50)


    
    
def predicting_model(df):
    
    data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)
    
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
    input_data = scaler.fit_transform(final_df)
    
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100 : i])
        y_test.append(input_data[i, 0])
        
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    model = load_model('keras_model.h5')
    
    y_predicted = model.predict(x_test)
    
    scale_factor = 1/float(scaler.scale_)
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    y_predicted = np.array(y_predicted).flatten()
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df['Date'], y = y_test, name="Original Price"))
    fig5.add_trace(go.Scatter(x=df['Date'], y = y_predicted, name="Predicted Price"))
    fig5.layout.update(title_text='Original vs Predicted Prices', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig5)
    
# Data Collection and Preprocessing

yf.pdr_override()

st.title("Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker", 'TSLA')

start = st.date_input("Start", value = pd.to_datetime('2021-01-01'))
end = st.date_input("End", value = pd.to_datetime('today'))

dataframe = pdr.get_data_yahoo(user_input, start, end)
dataframe.reset_index(inplace = True)
data = dataframe.drop(dataframe.columns[[0, 2, 3, 6]], axis = 1)

Descriptive_Stats(dataframe, data)
Prob_Distributions(dataframe, data)
Hypothesis_Test(dataframe, data)
plot_raw_data(dataframe)
plot_rateofreturn(dataframe)
plot_moving_averages(dataframe)
predicting_model(dataframe)