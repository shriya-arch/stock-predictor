import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def generate_gbm_prices(n_days=252, mu=0.0003, sigma=0.015, S0=18000, seed=42):
    np.random.seed(seed)
    dW = np.random.normal(0, 1, n_days)
    prices = np.zeros(n_days)
    prices[0] = S0
    for t in range(1, n_days):
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * dW[t])
    df = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=n_days, freq="B"),
        "Price": prices
    }).set_index("Date")
    df["Returns"] = np.diff(prices, prepend=prices[0]) / prices
    df["LogReturns"] = np.log(prices / np.roll(prices, 1))
    df["LogReturns"].iloc[0] = 0
    return df

def engineer_features(df, macro):
    f = df.copy()
    for lag in [1,3,7,14,30]:
        f[f"lag_{lag}"] = f["Price"].shift(lag)
    for w in [5,10,20,30]:
        f[f"roll_mean_{w}"] = f["Price"].rolling(w).mean()
        f[f"roll_std_{w}"] = f["Price"].rolling(w).std()
    f["volatility_21d"] = f["LogReturns"].rolling(21).std() * np.sqrt(252)
    for k,v in macro.items():
        f[k] = v
    return f.dropna()

def label_crashes(df, threshold=-0.1, horizon=10):
    p = df["Price"].values
    crash = np.zeros(len(p))
    for i in range(len(p)-horizon):
        if (p[i+1:i+1+horizon].min() - p[i]) / p[i] <= threshold:
            crash[i] = 1
    df["Crash"] = crash
    return df

def build_seq(series, w=30):
    X,y = [],[]
    for i in range(w,len(series)):
        X.append(series[i-w:i])
        y.append(series[i])
    return np.array(X).reshape(-1,w,1), np.array(y)

def build_lstm(w):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(w,1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def train_lstm(prices, w=30, epochs=30):
    scaler = MinMaxScaler()
    s = scaler.fit_transform(prices.reshape(-1,1)).flatten()
    X,y = build_seq(s,w)
    split = int(len(X)*0.8)
    m = build_lstm(w)
    m.fit(X[:split], y[:split], epochs=epochs, batch_size=16,
          validation_split=0.1, callbacks=[EarlyStopping(patience=5)], verbose=0)
    preds = scaler.inverse_transform(m.predict(X).reshape(-1,1)).flatten()
    return preds, split+w

def train_xgb(df, lstm_preds):
    n = min(len(df), len(lstm_preds))
    df = df.iloc[-n:]
    y = df["Price"].values
    X = df.drop(columns=["Price","Returns","LogReturns","Crash"], errors="ignore")
    X["lstm"] = lstm_preds[-n:]
    res = y - lstm_preds[-n:]
    split = int(n*0.8)
    model = XGBRegressor(n_estimators=150, max_depth=4)
    model.fit(X[:split], res[:split])
    final = lstm_preds[-n:] + model.predict(X)
    return final, model, X

def metrics(a,p):
    return {
        "RMSE": np.sqrt(mean_squared_error(a,p)),
        "MAE": mean_absolute_error(a,p),
        "MAPE": np.mean(np.abs((a-p)/(a+1e-8)))*100
    }

def crash_prob(macro, r):
    vol = r.std()*np.sqrt(252)
    mom = r[-5:].mean()
    s = (0.3*min(vol,1)+0.25*macro["vix"]/80+
         0.2*macro["inflation"]/12+0.15*macro["interest_rate"]/10+
         0.1*max(0,-mom*200))
    return 1/(1+np.exp(-8*(s-0.4)))

st.set_page_config(layout="wide")
st.title("📈 Stock Predictor")

with st.sidebar:
    n = st.slider("Days",100,500,252)
    w = st.slider("Window",10,60,30)
    e = st.slider("Epochs",5,50,20)
    vix = st.slider("VIX",10,80,20)
    inf = st.slider("Inflation",1.0,10.0,4.0)
    rate = st.slider("Rate",1.0,10.0,5.0)
    macro = {"vix":vix,"inflation":inf,"interest_rate":rate}

if st.button("Run"):
    df = generate_gbm_prices(n_days=n)
    df = label_crashes(df)
    feat = engineer_features(df, macro)

    lstm_preds, idx = train_lstm(df["Price"].values, w, e)
    final_preds, model, X = train_xgb(feat, lstm_preds)

    actual = feat["Price"].values[-len(final_preds):]
    m = metrics(actual, final_preds)
    prob = crash_prob(macro, df["Returns"].values[-30:])

    st.metric("Crash Probability", f"{prob*100:.2f}%")
    st.metric("RMSE", f"{m['RMSE']:.2f}")

    fig, ax = plt.subplots()
    ax.plot(df["Price"].values, label="Actual")
    ax.plot(range(len(df)-len(final_preds), len(df)), final_preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    shap_vals = shap.TreeExplainer(model).shap_values(X)
    shap.summary_plot(shap_vals, X, show=False)
    