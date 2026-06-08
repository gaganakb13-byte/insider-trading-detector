import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import smtplib
from email.mime.text import MIMEText

# Title
st.title("AI-Powered Insider Trading Detector")

# --- Kaggle Dataset Section ---
try:
    kaggle_data = pd.read_csv("market_indicators.csv")
    st.subheader("Kaggle Dataset Preview")
    st.write(kaggle_data.head())
    st.write("Columns in Kaggle dataset:", kaggle_data.columns)

    # Volatility calculation depending on available columns
    if "High" in kaggle_data.columns and "Low" in kaggle_data.columns:
        kaggle_data['Volatility'] = kaggle_data['High'] - kaggle_data['Low']
    elif "Open" in kaggle_data.columns and "Close" in kaggle_data.columns:
        kaggle_data['Volatility'] = kaggle_data['Close'] - kaggle_data['Open']
    else:
        kaggle_data['Volatility'] = 0

    # If Volume exists, use it; else skip
    if "Volume" in kaggle_data.columns:
        avg_volume_kaggle = kaggle_data['Volume'].mean()
        kaggle_data['Volume_Anomaly'] = kaggle_data['Volume'] > (avg_volume_kaggle * 2)
    else:
        kaggle_data['Volume_Anomaly'] = False

    avg_volatility_kaggle = kaggle_data['Volatility'].mean()
    kaggle_data['Volatility_Anomaly'] = kaggle_data['Volatility'] > (avg_volatility_kaggle * 2)

    kaggle_data['Suspicion_Score'] = 0
    kaggle_data.loc[kaggle_data['Volume_Anomaly'], 'Suspicion_Score'] += 50
    kaggle_data.loc[kaggle_data['Volatility_Anomaly'], 'Suspicion_Score'] += 50

    flagged_kaggle = kaggle_data[kaggle_data['Suspicion_Score'] > 0]
    st.subheader("Flagged Suspicious Days (Kaggle Data)")
    st.write(flagged_kaggle)

    # Plot
    st.subheader("Kaggle Volatility vs Suspicion")
    fig3, ax3 = plt.subplots()
    ax3.plot(kaggle_data.index, kaggle_data['Volatility'], label="Volatility", color="orange")
    ax3.scatter(flagged_kaggle.index, flagged_kaggle['Volatility'], color='red', label="Suspicious")
    ax3.legend()
    st.pyplot(fig3)

    # --- Machine Learning on Kaggle ---
    st.subheader("Machine Learning Prediction (Random Forest)")
    features = []
    for col in ["Open", "Close", "Volume", "Volatility"]:
        if col in kaggle_data.columns:
            features.append(col)

    if len(features) >= 2:
        X = kaggle_data[features]
        y = (kaggle_data['Suspicion_Score'] > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Not enough features for ML model.")

except Exception as e:
    st.error(f"Could not load Kaggle dataset: {e}")

# --- Yahoo Finance Section ---
ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, INFY.NS)", "RELIANCE.NS")
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])

if ticker:
    data = yf.download(ticker, period=period, interval="1d")
    st.subheader(f"Stock Data for {ticker}")
    st.write(data.tail())

    avg_volume = data['Volume'].mean()
    data['Volatility'] = data['High'] - data['Low']
    avg_volatility = data['Volatility'].mean()

    data['Volume_Anomaly'] = data['Volume'] > (avg_volume * 2)
    data['Volatility_Anomaly'] = data['Volatility'] > (avg_volatility * 2)

    data['Suspicion_Score'] = 0
    data.loc[data['Volume_Anomaly'], 'Suspicion_Score'] += 50
    data.loc[data['Volatility_Anomaly'], 'Suspicion_Score'] += 50

    flagged = data[data['Suspicion_Score'] > 0]
    st.subheader("Flagged Suspicious Trading Days (Yahoo Finance)")
    st.write(flagged[['Open','Close','Volume','Volatility','Suspicion_Score']])

    # Plots
    st.subheader("Trading Volume vs Suspicion")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Volume'], label="Volume")
    ax.scatter(flagged.index, flagged['Volume'], color='red', label="Suspicious")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Daily Volatility vs Suspicion")
    fig2, ax2 = plt.subplots()
    ax2.plot(data.index, data['Volatility'], label="Volatility", color="orange")
    ax2.scatter(flagged.index, flagged['Volatility'], color='red', label="Suspicious")
    ax2.legend()
    st.pyplot(fig2)

    # Sentiment Analysis
    headlines = [
        "Company reports record profits",
        "CEO under investigation for fraud",
        "Market shows signs of volatility"
    ]
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]

    st.subheader("News Sentiment Analysis")
    for h, s in zip(headlines, sentiments):
        st.write(f"{h} → Sentiment Score: {s:.2f}")

    # --- Email Alert System ---
    if st.button("Send Email Alert"):
        try:
            msg = MIMEText(flagged.to_string())
            msg['Subject'] = f"Suspicious Trading Alert for {ticker}"
            msg['From'] = "gaganakb203@gmail.com"
            msg['To'] = "lishalokesh1208@gmail.com"

            # Gmail SMTP example (replace with your credentials)
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login("gaganakb203@gmail.com", "zfgjqarfsnnjuasx")
                server.send_message(msg)

            st.success("Email alert sent successfully!")
        except Exception as e:
            st.error(f"Email sending failed: {e}")
