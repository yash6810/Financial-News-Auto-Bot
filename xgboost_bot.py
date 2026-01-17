import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os

# --- Helper Functions for Technical Indicators ---
# PERFORMANCE OPTIMIZATION: These functions use vectorized pandas operations for speed.
def compute_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    # DATA TRANSFORMATION: Calculate price change.
    delta = data['Close'].diff()
    # DATA TRANSFORMATION: Separate gains and losses.
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # DATA TRANSFORMATION: Calculate RSI.
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    # DATA TRANSFORMATION: Calculate short-term and long-term exponential moving averages.
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    # DATA TRANSFORMATION: Calculate MACD line and signal line.
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# --- Data Fetching and Feature Engineering ---
# CONFIGURATION LOADING: Hardcoded ticker. Consider moving "TCS.NS" to a config file.
TICKER = "TCS.NS"
# EXTERNAL API CALL: Fetching historical stock data from Yahoo Finance.
# ERROR HANDLING: Wrap in a try...except block to handle network issues or invalid tickers.
try:
    data = yf.download(TICKER, start="2020-01-01", interval="1d")
    # INPUT VALIDATION: Ensure that the downloaded data is not empty.
    if data.empty:
        print(f"No data downloaded for {TICKER}. Exiting.")
        exit()
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()


# Flatten multi-index columns from yfinance, e.g., ('Close', 'TCS.NS') -> 'Close'
# DATA TRANSFORMATION: Simplifies column names for easier access.
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. CALCULATE ALL INDICATORS
# DATA TRANSFORMATION: Calculate daily returns.
data['Returns'] = data['Close'].pct_change()
# DATA TRANSFORMATION: Calculate Simple Moving Averages (50 and 200 days).
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
# DATA TRANSFORMATION: Calculate Relative Strength Index (RSI).
data['RSI'] = compute_rsi(data)
# DATA TRANSFORMATION: Calculate MACD and its signal line.
data['MACD'], data['Signal_Line'] = compute_macd(data)

# DATA TRANSFORMATION: Calculate Bollinger Bands.
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Std_20'] = data['Close'].rolling(window=20).std()
data['Upper_Band'] = data['SMA_20'] + (2 * data['Std_20'])
data['Lower_Band'] = data['SMA_20'] - (2 * data['Std_20'])
# DATA TRANSFORMATION: Calculate Bollinger Band position.
data['BB_Position'] = (data['Close'] - data['Lower_Band']) / (data['Upper_Band'] - data['Lower_Band'])

# DATA TRANSFORMATION: Create lagged returns for previous days.
for i in range(1, 4):
    data[f'Returns_{i}'] = data['Returns'].shift(i)

# 3. CLEAN DATA
# DATA TRANSFORMATION: Remove rows with NaN values resulting from indicator calculations.
data.dropna(inplace=True)

# --- Model Training and Evaluation for Different Horizons ---
horizons = [1, 3, 5]
best_accuracy = 0
best_horizon = 0
best_model = None
best_test_data = None

for horizon in horizons:
    print(f"\n--- Testing Prediction Horizon: {horizon} Day(s) ---")

    # DATA TRANSFORMATION: Create the target variable for classification.
    # The target is 1 if the price increases after 'horizon' days, otherwise 0.
    temp_data = data.copy()
    temp_data['Target'] = np.where(temp_data['Close'].shift(-horizon) > temp_data['Close'], 1, 0)
    temp_data = temp_data.iloc[:-horizon]

    # 5. SPLIT DATA
    # DATA TRANSFORMATION: Split data into training and testing sets.
    train = temp_data.iloc[:-250]
    test = temp_data.iloc[-250:]

    features = [
        'SMA_50', 'SMA_200', 'RSI', 'Returns', 'MACD', 'Signal_Line',
        'BB_Position', 'Returns_1', 'Returns_2', 'Returns_3'
    ]
    X_train = train[features]
    y_train = train['Target']
    X_test = test[features]
    y_test = test['Target']

    # --- Model Training and Evaluation ---
    # 1. HANDLE CLASS IMBALANCE
    # BUSINESS LOGIC: Calculate weight to handle imbalanced classes in the target variable.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 2. TRAIN XGBOOST MODEL (using best params from previous grid search)
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        gamma=0.2,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # 3. MAKE PREDICTIONS
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy for {horizon}-Day Horizon: {acc*100:.2f}%")

    # BUSINESS LOGIC BRANCH: Select the model with the highest accuracy.
    if acc > best_accuracy:
        best_accuracy = acc
        best_horizon = horizon
        best_model = model
        best_test_data = test.copy()
        best_predictions = predictions

print(f"\nBest performing horizon is {best_horizon} day(s) with an accuracy of {best_accuracy*100:.2f}%")

# --- 4. INTEGRATING MARKET SENTIMENT ---
print("--- 4. INTEGRATING MARKET SENTIMENT ---")

# CONFIGURATION LOADING: Load external sentiment score from a JSON file.
# RESOURCE MANAGEMENT: 'with open' ensures the file is properly closed.
sentiment_score = 0.0 # Default to neutral
# ERROR HANDLING: Check if sentiment file exists.
if os.path.exists("market_mood.json"):
    try:
        with open("market_mood.json", "r") as f:
            mood_data = json.load(f)
            sentiment_score = mood_data.get("mood_score", 0)
            print(f"   >> MARKET SENTIMENT LOADED: {sentiment_score:.4f}")
    except json.JSONDecodeError:
        print("   >> WARNING: 'market_mood.json' is corrupted. Assuming Neutral (0.0).")
    except Exception as e:
        print(f"   >> WARNING: Error loading 'market_mood.json': {e}. Assuming Neutral (0.0).")
else:
    print("   >> WARNING: No sentiment file found. Assuming Neutral (0.0).")


# BUSINESS LOGIC BRANCH: Adjust trading strategy based on market sentiment.
# This acts as a filter on the AI's decisions.
buy_threshold = 0.50  # Standard AI confidence level

# SECURITY CHECK: If sentiment is negative, engage defensive mode to prevent risky trades.
if sentiment_score < -0.05:
    print("   >> ALERT: Market News is NEGATIVE. Engaging Defensive Mode.")
    print("   >> ACTION: Buying is DISABLED or Restricted.")
    buy_threshold = 0.99 # Set threshold to a near-impossible level to prevent buys.
# BUSINESS LOGIC BRANCH: If sentiment is positive, use standard trading parameters.
elif sentiment_score > 0.05:
    print("   >> ALERT: Market News is POSITIVE.")
    print("   >> ACTION: Standard Trading Enabled.")
    buy_threshold = 0.50
# BUSINESS LOGIC BRANCH: If sentiment is neutral, be slightly more cautious.
else:
    print("   >> ALERT: Market News is NEUTRAL.")
    buy_threshold = 0.55 # Slightly stricter threshold for neutral sentiment.

# 3. Predict with the Filter
# Get probabilities (confidence) instead of just 0/1
features = [
    'SMA_50', 'SMA_200', 'RSI', 'Returns', 'MACD', 'Signal_Line',
    'BB_Position', 'Returns_1', 'Returns_2', 'Returns_3'
]
X_test_best = best_test_data[features]
probs = best_model.predict_proba(X_test_best)[:, 1]

# Apply the Filter
# DATA TRANSFORMATION: Convert probabilities to final buy/sell signals based on the sentiment-adjusted threshold.
final_signals = (probs > buy_threshold).astype(int)


# --- Backtesting with Best Performing Model ---
print(f"\n--- Backtesting with Best Model (Horizon: {best_horizon} days) ---")
# CONFIGURATION LOADING: Hardcoded backtesting parameters. Consider moving to a config file.
capital = 100000
commission = 0.001
shares = 0
in_position = False
prices = best_test_data['Close'].values

for i in range(len(final_signals)):
    # BUSINESS LOGIC BRANCH: Execute a 'BUY' if signal is 1 and not already in a position.
    if final_signals[i] == 1 and not in_position: # BUY
        capital_after_commission = capital * (1 - commission)
        shares = capital_after_commission / prices[i]
        capital = 0
        in_position = True
        print(f"BUY: {shares:.2f} shares at ₹{prices[i]:.2f}")
    # BUSINESS LOGIC BRANCH: Execute a 'SELL' if signal is 0 and currently in a position.
    elif final_signals[i] == 0 and in_position: # SELL
        sale_value = shares * prices[i]
        capital = sale_value * (1 - commission)
        shares = 0
        in_position = False
        print(f"SELL: at ₹{prices[i]:.2f}, Capital: ₹{capital:,.2f}")

# Final Value Calculation
# BUSINESS LOGIC BRANCH: If still in a position at the end, calculate final capital based on the last price.
if in_position:
    sale_value = shares * prices[-1]
    final_capital = sale_value * (1 - commission)
else:
    final_capital = capital

profit = final_capital - 100000
print(f"\nInitial Capital: ₹100,000")
print(f"Final Capital: ₹{final_capital:,.2f}")
print(f"Profit: ₹{profit:,.2f}")

# --- Task 3.1: Calculate "Buy & Hold" Risk ---
print("\n--- Buy & Hold Risk Analysis (2024) ---")

# Filter data for the year 2024
# The data is fetched from 2020, so 2024 data should be available.
data_2024 = data.loc[data.index.year == 2024].copy()

# INPUT VALIDATION: Check if there is data for the year 2024.
if not data_2024.empty:
    # Find the absolute peak in 2024
    peak_price_2024 = data_2024['Close'].max()
    peak_date_2024 = data_2024['Close'].idxmax()

    # Find the trough price that occurs after the peak
    trough_price_2024 = data_2024.loc[peak_date_2024:]['Close'].min()

    # DATA TRANSFORMATION: Calculate Max Drawdown for the Buy & Hold strategy.
    max_drawdown_bh = (peak_price_2024 - trough_price_2024) / peak_price_2024

    print(f"TCS Stock Analysis for Buy & Hold in 2024:")
    print(f"  - Peak Price: ₹{peak_price_2024:,.2f} on {peak_date_2024.date()}")
    print(f"  - Trough Price after Peak: ₹{trough_price_2024:,.2f}")
    print(f"Max Drawdown: {(max_drawdown_bh * 100):.2f}%")
else:
    # ERROR HANDLING: Message when no data is available for the analysis.
    print("No data available for 2024 to calculate 'Buy & Hold' risk.")