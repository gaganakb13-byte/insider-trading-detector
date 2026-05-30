import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("AI-Based Insider Trading Detector")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, INFY.NS)", "RELIANCE.NS")
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])

if ticker:
    # Fetch data
    data = yf.download(ticker, period=period, interval="1d")
    st.subheader(f"Stock Data for {ticker}")
    st.write(data.tail())

    # Calculate baseline averages
    avg_volume = data['Volume'].mean()
    data['Volatility'] = data['High'] - data['Low']
    avg_volatility = data['Volatility'].mean()

    # Detect anomalies
    data['Volume_Anomaly'] = data['Volume'] > (avg_volume * 2)
    data['Volatility_Anomaly'] = data['Volatility'] > (avg_volatility * 2)

    # Suspicion Score
    data['Suspicion_Score'] = 0
    data.loc[data['Volume_Anomaly'], 'Suspicion_Score'] += 50
    data.loc[data['Volatility_Anomaly'], 'Suspicion_Score'] += 50

    # Show flagged events
    flagged = data[data['Suspicion_Score'] > 0]
    st.subheader("Flagged Suspicious Trading Days")
    st.write(flagged[['Open','Close','Volume','Volatility','Suspicion_Score']])

    # Plot Volume anomalies
    st.subheader("Trading Volume vs Suspicion")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Volume'], label="Volume")
    ax.scatter(flagged.index, flagged['Volume'], color='red', label="Suspicious")
    ax.legend()
    st.pyplot(fig)

    # Plot Volatility anomalies
    st.subheader("Daily Volatility vs Suspicion")
    fig2, ax2 = plt.subplots()
    ax2.plot(data.index, data['Volatility'], label="Volatility", color="orange")
    ax2.scatter(flagged.index, flagged['Volatility'], color='red', label="Suspicious")
    ax2.legend()
    st.pyplot(fig2)
