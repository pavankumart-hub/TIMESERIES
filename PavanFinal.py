# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 PAVAN-HYBRID ARIMA Stock Forecast (stable)", layout="wide")
st.title("Polynomial Regression + ARIMA Stock Forecast")
st.markdown("Live your Life as an Exclamation rather than an Explanation-SIR ISSAC NEWTON")
st.markdown("True perspective of God's creation lies in the Art of understanding Mathematics-PAVAN KUMAR THOTA")
st.markdown("Earning in the face of Risk-STOCK MARKET")
st.markdown("Tests: ADF, KPSS, PP, Jarque-Bera, L-jung Box")

# Sidebar inputs
st.sidebar.header("INPUT-ARIMA ORIGINAL")
ticker = st.sidebar.text_input("Stock Ticker", "TATASTEEL.NS").upper()

# Calendar date selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date",
                              value=datetime(2008, 1, 1).date(),
                              min_value=datetime(2000, 1, 1).date(),
                              max_value=datetime.now().date())
with col2:
    end_date = st.date_input("End Date",
                            value=datetime.now().date(),
                            min_value=datetime(2000, 1, 1).date(),
                            max_value=datetime.now().date())

# Price type selection
price_type = st.sidebar.selectbox("Select Dependent Price Type (Y)", ["High", "Low", "Open", "Close", "Adj Close"])
price_type1 = st.sidebar.selectbox("Select Independent Price Type (X)", ["High", "Low", "Open", "Close", "Adj Close"])
st.sidebar.header("Forecast Input")
today_open_input = st.sidebar.number_input(f"Enter Today's Open Price",
                                       value=100.0,
                                       min_value=0.0,
                                       step=0.1,
                                       key="today_open_input")
degree = st.sidebar.slider("Polynomial Degree", 1, 20, 3)

# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p_range = st.sidebar.slider("P (AR) Range", 0, 5, (0, 2))
q_range = st.sidebar.slider("Q (MA) Range", 0, 5, (0, 2))
d_range = st.sidebar.slider("D (Differencing) Range", 0, 2, (0, 1))
run_analysis_btn = st.sidebar.button("Run Complete Analysis", type="primary")
