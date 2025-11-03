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

# Function to detect currency based on ticker
def detect_currency(ticker):
    # Indian stock indicators
    indian_indicators = ['.NS', '.BO', '.NSE', '.BSE', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK',
                         'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'ITC', 'LT', 'BHARTIARTL']

    # Check if ticker contains Indian indicators
    if any(indicator in ticker.upper() for indicator in indian_indicators):
        return "₹"
    else:
        return "$"

def safe_ljungbox(resids, max_lag=10):
    n = len(resids)
    # choose a safe lag: at most n-1 and at most requested max_lag
    lag = min(max_lag, max(1, n - 1))
    try:
        result = acorr_ljungbox(resids, lags=[lag], return_df=True)
        pval = float(result["lb_pvalue"].iloc[0])
        stat = float(result["lb_stat"].iloc[0]) if "lb_stat" in result.columns else float(result["lb_value"].iloc[0])
        return stat, pval, None
    except Exception as ex:
        return None, None, str(ex)

def safe_jarque_bera(resids):
    try:
        jb_stat, jb_p = jarque_bera(resids)
        return float(jb_stat), float(jb_p), None
    except Exception as ex:
        return None, None, str(ex)

def safe_kpss(data):
    try:
        # KPSS test with different regression types
        kpss_stat, p_value, lags, critical_values = kpss(data, regression='ct', nlags='auto')

        # Manual p-value calculation based on critical values
        cv_1pct = critical_values['1%']
        cv_5pct = critical_values['5%']
        cv_10pct = critical_values['10%']

        # Determine p-value based on test statistic and critical values
        if kpss_stat > cv_1pct:
            manual_pvalue = 0.01
        elif kpss_stat > cv_5pct:
            manual_pvalue = 0.05
        elif kpss_stat > cv_10pct:
            manual_pvalue = 0.10
        else:
            manual_pvalue = 0.50

        return float(kpss_stat), float(manual_pvalue), critical_values, None
    except Exception as ex:
        return None, None, None, str(ex)

def safe_adfuller(data):
    try:
        adf_result = adfuller(data)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
        return adf_stat, adf_p, None
    except Exception as ex:
        return None, None, str(ex)

def adf_test(data):
    try:
        adf_result = adfuller(data)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
        return adf_stat, adf_p, None
    except Exception as ex:
        return None, None, f"ADF test failed: {str(ex)}"


def fit_arima_model(data, p, d, q):
    try:
        model = SARIMAX(data, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
        fitted_model = model.fit(disp=False)
        return fitted_model, None
    except Exception as ex:
        return None, str(ex)

if run_analysis_btn:
        st.header(f"Complete Analysis for {ticker}")

        # Auto-detect currency
        currency_symbol = detect_currency(ticker)
        st.sidebar.info(f"Detected Currency: {currency_symbol}")

        with st.spinner(f"Downloading {ticker}..."):
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data is None or data.empty:
                st.error("No data found. Check ticker symbol or date range.")
                st.stop()

        # Check if both price columns exist
        if price_type not in data.columns:
                st.error(f"Price type '{price_type}' not found in data. Available columns: {list(data.columns)}")
                st.stop()

        if price_type1 not in data.columns:
                st.error(f"Price type '{price_type1}' not found in data. Available columns: {list(data.columns)}")
                st.stop()

        # Extract and clean both price data series
        price_data = data[price_type].copy()
        price_data = price_data.dropna()

        price_data1 = data[price_type1].copy()
        price_data1 = price_data1.dropna()

        # Ensure both series have the same length after dropping NaN values
        # Align both series to have the same dates
        common_dates = price_data.index.intersection(price_data1.index)
        price_data = price_data.loc[common_dates]
        price_data1 = price_data1.loc[common_dates]

        n = len(price_data)
        if n < 10:
                st.error(f"Not enough common data points after cleaning ({n}). Increase date range or pick another ticker.")
                st.stop()

        st.success(f"Successfully loaded {n} data points for both price series")
        # show basics - FIXED: Extract scalar values for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_price = float(price_data.iloc[-1])
            st.metric(f"Current {price_type} Price", f"{currency_symbol}{current_price:.2f}")
        with col2:
            st.metric("Data Points", n)
        with col3:
            actual_days = (price_data.index[-1] - price_data.index[0]).days
            st.metric("Analysis Period (days)", actual_days)
        # KPSS Test on original data
        st.subheader("KPSS Test - Stationarity Check (Original Data)")
        kpss_stat, kpss_p, kpss_critical_values, kpss_err = safe_kpss(price_data)

        if kpss_err:
            st.error(f"KPSS test error: {kpss_err}")
        else:
            st.write(f"**KPSS Test Statistic:** {kpss_stat:.6f}")
            st.write(f"**5% Critical Value:** {kpss_critical_values['5%']:.4f}")

            if kpss_stat > kpss_critical_values['5%']:
                st.error("✗ Data is Difference-stationary (test statistic > 5% critical value)")
            else:
                st.success("✓ Data appears Trend-stationary (test statistic < 5% critical value)")
        # Prepare X: center + scale ordinal dates


        # Date preprocessing and normalization
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean(axis=0)[0])
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min

        if dates_range == 0:
            st.error("All dates identical (unexpected).")
            st.stop()

        X = (dates - dates_mean) / dates_range
        y = price_data.values.astype(float)
        # Polynomial regression model
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Forecast next day value
        last_normalized_date = X[-1][0]  # Get last normalized date
        next_normalized_date = last_normalized_date + (1 / dates_range)  # Add one day in normalized scale

        # Create next day's features and apply polynomial transformation
        next_day_features = np.array([[next_normalized_date]])
        next_day_poly = poly.transform(next_day_features)

        # Predict next day value
        next_day_prediction = model.predict(next_day_poly)

        # Convert to scalar values to avoid numpy array formatting issues
        current_price = float(y[-1]) if hasattr(y[-1], '__iter__') else y[-1]
        predicted_price = float(next_day_prediction[0])
        price_change = predicted_price - current_price
        percent_change = (price_change / current_price) * 100

        # Streamlit display
        st.subheader("📈 Next Day Forecast")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Current Price",
                value=f"{currency_symbol}{current_price:.2f}"
            )
        with col2:
            st.metric(
                label="Predicted Price",
                value=f"{currency_symbol}{predicted_price:.2f}",
                delta=f"{price_change:.2f}"
            )

        # Additional forecast details
        with st.expander("Forecast Details"):
            st.write(f"**Model:** Polynomial Regression (Degree {degree})")
            st.write(f"**Date Range:** {dates_range:.0f} days")
            st.write(f"**Last available date:** {price_data.index[-1].strftime('%Y-%m-%d')}")

            # Calculate next actual date
            next_actual_date = price_data.index[-1] + pd.Timedelta(days=1)
            st.write(f"**Forecast date:** {next_actual_date.strftime('%Y-%m-%d')}")

            st.write(f"**Price change:** {currency_symbol}{price_change:.2f}")
            st.write(f"**Percent change:** {percent_change:.2f}%")

        # Optional: Show prediction confidence or model performance
        st.info(f"*Forecast based on polynomial regression model with degree {degree}*")

        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean(axis=0)[0])
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min
        if dates_range == 0:
            st.error("All dates identical (unexpected).")
            st.stop()
        X = (dates - dates_mean) / dates_range

        y = price_data.values.astype(float)

        # Build polynomial features and fit model
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        # Metrics - FIXED: Ensure we're using scalar values
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        r2 = float(r2_score(y, y_pred))

        st.subheader("Polynomial Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{currency_symbol}{rmse:.4f}")
        c2.metric("R²", f"{r2:.4f}")

        # Plot actual vs predicted
        st.subheader(f"Actual vs Predicted (Degree = {degree})")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(price_data.index, y, label=f"Actual {price_type}", linewidth=2)
        ax.plot(price_data.index, y_pred, label="Predicted", linestyle="--", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Price ({currency_symbol})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Residuals
        residuals = y - y_pred
        st.subheader("Residual Analysis")

        # Residual time plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(price_data.index, residuals, label="Residuals",color="red")
        ax.axhline(0, linestyle="--", color="k")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Residual ({currency_symbol})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # ACF and PACF plots for residuals
        st.subheader("ACF and PACF Plots for Residuals")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(residuals, ax=ax1, lags=20, color="green")
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(residuals, ax=ax2, lags=20,color="orange")
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        plt.tight_layout()
        st.pyplot(fig)

        # NEW: Residual Statistical Tests before ARIMA
        st.header("Residual Statistical Tests")

        # ADF Test for Residuals
        st.subheader("ADF Test - Stationarity Check (Residuals)")
        adf_stat, adf_p, adf_err = safe_adfuller(residuals)

        if adf_err:
            st.error(f"ADF test error: {adf_err}")
        else:
            st.write(f"**ADF Test Statistic:** {adf_stat:.6f}")
            st.write(f"**ADF p-value:** {adf_p:.6f}")

            if adf_p <= 0.05:
                st.success("✓ Residuals are Stationary (p-value ≤ 0.05)")
            else:
                st.error("✗ Residuals are Non-Stationary (p-value > 0.05)")

        # Residual Histogram with Skewness and Kurtosis
        st.subheader("Residual Distribution Analysis")

        # Calculate statistics - FIXED: Ensure scalar values
        residual_skew = float(skew(residuals))
        residual_kurtosis = float(kurtosis(residuals))
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{residual_mean:.6f}")
        with col2:
            st.metric("Std Dev", f"{residual_std:.6f}")
        with col3:
            st.metric("Skewness", f"{residual_skew:.4f}")
        with col4:
            st.metric("Kurtosis", f"{residual_kurtosis:.4f}")

        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        n_bins, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, color='red', edgecolor='black')

        # Add normal distribution curve for comparison
        from scipy.stats import norm
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, residual_mean, residual_std)
        ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

        ax.axvline(residual_mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {residual_mean:.4f}')
        ax.set_xlabel(f'Residual Value ({currency_symbol})')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution Histogram')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Interpret skewness and kurtosis
        st.write("**Distribution Interpretation:**")
        if abs(residual_skew) < 0.5:
            st.write("✓ Skewness: Approximately symmetric (close to 0)")
        elif residual_skew > 0.5:
            st.write("↗️ Skewness: Right-skewed (positive skew)")
        else:
            st.write("↙️ Skewness: Left-skewed (negative skew)")

        if abs(residual_kurtosis) < 1:
            st.write("✓ Kurtosis: Approximately normal (close to 0)")
        elif residual_kurtosis > 1:
            st.write("📈 Kurtosis: Leptokurtic (heavy-tailed)")
        else:
            st.write("📉 Kurtosis: Platykurtic (light-tailed)")

        # Normality Test
        st.subheader("Normality Test (Jarque-Bera)")
        jb_stat, jb_p, jb_err = safe_jarque_bera(residuals)

        if jb_err:
            st.error(f"Jarque-Bera test error: {jb_err}")
        else:
            st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f}")
            st.write(f"**Jarque-Bera p-value:** {jb_p:.4f}")

            if jb_p > 0.05:
                st.success("✓ Residuals are Normally Distributed (p-value > 0.05)")
            else:
                st.error("✗ Residuals are NOT Normally Distributed (p-value ≤ 0.05)")

        # Autocorrelation Test (Ljung-Box)
        st.subheader("Autocorrelation Test (Ljung-Box)")
        lb_stat, lb_p, lb_err = safe_ljungbox(residuals, max_lag=10)

        if lb_err:
            st.error(f"Ljung-Box test error: {lb_err}")
        else:
            st.write(f"**Ljung-Box Statistic:** {lb_stat:.4f}")
            st.write(f"**Ljung-Box p-value:** {lb_p:.4f}")

            if lb_p > 0.05:
                st.success("✓ No Significant Autocorrelation (p-value > 0.05)")
            else:
                st.error("✗ Significant Autocorrelation Present (p-value ≤ 0.05)")

