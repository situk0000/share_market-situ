from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# Initialize FastAPI

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Config

CHART_HISTORY_DAYS = 60 


# Load Models and Scalers
# --------------------------
# Load ADANIPORTS model (default/fallback)
lstm_model_adani = load_model("lstm_close_model.h5", compile=False)
scaler_adani = joblib.load("lstm_scaler.pkl")

# Try to load BSE-specific model, otherwise use ADANI model
try:
    lstm_model_bse = load_model("bse_model.h5", compile=False)
    scaler_bse = joblib.load("bse_scaler.pkl")
    print("✓ BSE model loaded successfully")
except:
    lstm_model_bse = lstm_model_adani
    scaler_bse = scaler_adani
    print("⚠ BSE model not found, using ADANIPORTS model")

# Try to load NIFTY-specific model, otherwise use ADANI model
try:
    lstm_model_nifty = load_model("nifty_model.h5", compile=False)
    scaler_nifty = joblib.load("nifty_scaler.pkl")
    print("✓ NIFTY model loaded successfully")
except:
    lstm_model_nifty = lstm_model_adani
    scaler_nifty = scaler_adani
    print("⚠ NIFTY model not found, using ADANIPORTS model")

# --------------------------
# Load Historical Data
# --------------------------
# ADANIPORTS data
df_adani = pd.read_csv("ADANIPORTS.csv", parse_dates=["Date"])
df_adani = df_adani.sort_values("Date").reset_index(drop=True)
print("✓ ADANIPORTS.csv loaded successfully")

# BSE data - with column standardization
try:
    df_bse = pd.read_csv("BSE.csv")
    
    # Standardize all column names (any case to Title case)
    column_mapping = {}
    for col in df_bse.columns:
        col_lower = col.lower().strip()
        if col_lower == 'date':
            column_mapping[col] = 'Date'
        elif col_lower == 'open':
            column_mapping[col] = 'Open'
        elif col_lower == 'high':
            column_mapping[col] = 'High'
        elif col_lower == 'low':
            column_mapping[col] = 'Low'
        elif col_lower == 'close':
            column_mapping[col] = 'Close'
        elif col_lower == 'volume':
            column_mapping[col] = 'Volume'
    
    # Rename columns if mapping exists
    if column_mapping:
        df_bse = df_bse.rename(columns=column_mapping)
    
    # Convert Date to datetime and sort
    df_bse['Date'] = pd.to_datetime(df_bse['Date'])
    df_bse = df_bse.sort_values("Date").reset_index(drop=True)
    
    print("✓ BSE.csv loaded successfully")
    print(f"  Columns: {df_bse.columns.tolist()}")
except FileNotFoundError:
    print("⚠ BSE.csv not found! Please add BSE SENSEX data")
    df_bse = df_adani.copy()

# NIFTY data - with column standardization
try:
    df_nifty = pd.read_csv("NIFTY50.csv")
    
    # Standardize all column names (lowercase to Title case)
    column_mapping = {}
    for col in df_nifty.columns:
        col_lower = col.lower().strip()
        if col_lower == 'date':
            column_mapping[col] = 'Date'
        elif col_lower == 'open':
            column_mapping[col] = 'Open'
        elif col_lower == 'high':
            column_mapping[col] = 'High'
        elif col_lower == 'low':
            column_mapping[col] = 'Low'
        elif col_lower == 'close':
            column_mapping[col] = 'Close'
        elif col_lower == 'volume':
            column_mapping[col] = 'Volume'
    
    # Rename columns
    df_nifty = df_nifty.rename(columns=column_mapping)
    
    # Convert Date to datetime and sort
    df_nifty['Date'] = pd.to_datetime(df_nifty['Date'])
    df_nifty = df_nifty.sort_values("Date").reset_index(drop=True)
    
    print("✓ NIFTY50.csv loaded successfully")
    print(f"  Columns: {df_nifty.columns.tolist()}")
except FileNotFoundError:
    print("⚠ NIFTY50.csv not found! Please add NIFTY 50 data")
    df_nifty = df_adani.copy()


# --------------------------
# Forecast function
# --------------------------
def forecast_next_days_lstm(df, model, scaler, n_days=5, n_steps=10):
    """Forecast next N days using LSTM model"""
    close_prices = df["Close"].values
    forecasted = []
    last_sequence = close_prices[-n_steps:]
    
    current_date = pd.Timestamp.now()
    
    for day in range(1, n_days + 1):
        seq_scaled = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, n_steps, 1)
        pred_scaled = model.predict(seq_scaled, verbose=0)
        pred_close = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        
        forecast_date = current_date + pd.Timedelta(days=day)
        
        forecasted.append({
            "Date": forecast_date.strftime("%Y-%m-%d"),
            "pred_close": float(pred_close)
        })
        
        last_sequence = np.append(last_sequence[1:], pred_close)
    
    return forecasted

# Helper function for Chart Data
# --------------------------
def get_historical_chart_data(df: pd.DataFrame, days_of_history: int):
    """Extracts last N days of data for Chart.js"""
    df_recent = df.tail(days_of_history).copy()
    historical_dates = df_recent['Date'].dt.strftime('%Y-%m-%d').tolist()
    historical_prices = df_recent['Close'].tolist()
    return historical_dates, historical_prices

# --------------------------
# Routes - ADANIPORTS (NSE) - Home Page
# --------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    hist_dates, hist_prices = get_historical_chart_data(df_adani, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "forecast": None,
        "stock_name": "ADANIPORTS (NSE)",
        "last_close": float(df_adani["Close"].iloc[-1]),
        "last_date": df_adani["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "adaniports",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

@app.post("/forecast", response_class=HTMLResponse)
def forecast(request: Request, days: int = Form(...)):
    forecasted = forecast_next_days_lstm(df_adani, lstm_model_adani, scaler_adani, n_days=days)
    hist_dates, hist_prices = get_historical_chart_data(df_adani, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "forecast": forecasted,
        "stock_name": "ADANIPORTS (NSE)",
        "days": days,
        "last_close": float(df_adani["Close"].iloc[-1]),
        "last_date": df_adani["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "adaniports",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

# --------------------------
# Routes - BSE SENSEX
# --------------------------
@app.get("/bse", response_class=HTMLResponse)
def bse_page(request: Request):
    hist_dates, hist_prices = get_historical_chart_data(df_bse, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("BSE.html", {
        "request": request, 
        "forecast": None,
        "stock_name": "BSE SENSEX",
        "last_close": float(df_bse["Close"].iloc[-1]),
        "last_date": df_bse["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "bse",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

@app.post("/forecast_bse", response_class=HTMLResponse)
def forecast_bse(request: Request, days: int = Form(...)):
    # NOW USES BSE-SPECIFIC MODEL AND SCALER
    forecasted = forecast_next_days_lstm(df_bse, lstm_model_bse, scaler_bse, n_days=days)
    hist_dates, hist_prices = get_historical_chart_data(df_bse, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("BSE.html", {
        "request": request, 
        "forecast": forecasted,
        "stock_name": "BSE SENSEX",
        "days": days,
        "last_close": float(df_bse["Close"].iloc[-1]),
        "last_date": df_bse["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "bse",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

# --------------------------
# Routes - NIFTY 50
# --------------------------
@app.get("/nifty", response_class=HTMLResponse)
def nifty_page(request: Request):
    hist_dates, hist_prices = get_historical_chart_data(df_nifty, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("nifty.html", {
        "request": request, 
        "forecast": None,
        "stock_name": "NIFTY 50",
        "last_close": float(df_nifty["Close"].iloc[-1]),
        "last_date": df_nifty["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "nifty",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

@app.post("/forecast_nifty", response_class=HTMLResponse)
def forecast_nifty(request: Request, days: int = Form(...)):
    # NOW USES NIFTY-SPECIFIC MODEL AND SCALER
    forecasted = forecast_next_days_lstm(df_nifty, lstm_model_nifty, scaler_nifty, n_days=days)
    hist_dates, hist_prices = get_historical_chart_data(df_nifty, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("nifty.html", {
        "request": request, 
        "forecast": forecasted,
        "stock_name": "NIFTY 50",
        "days": days,
        "last_close": float(df_nifty["Close"].iloc[-1]),
        "last_date": df_nifty["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "nifty",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

# --------------------------
# API endpoint
# --------------------------
@app.get("/api/stocks")
def list_stocks():
    return {
        "stocks": [
            {"name": "ADANIPORTS", "endpoint": "/", "model": "lstm_close_model.h5"},
            {"name": "BSE SENSEX", "endpoint": "/bse", "model": "bse_model.h5 (or fallback)"},
            {"name": "NIFTY 50", "endpoint": "/nifty", "model": "nifty_model.h5 (or fallback)"}
        ],
        "note": "Each index uses its own model if available, otherwise falls back to ADANIPORTS model"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)