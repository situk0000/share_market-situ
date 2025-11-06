from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import sqlite3
import os
from contextlib import contextmanager

# ============================================================================
# Security Setup
# ============================================================================
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ============================================================================
# Database Setup
# ============================================================================
DB_PATH = "users.db"

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    """Database context manager"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ============================================================================
# Authentication Functions
# ============================================================================
def hash_password(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(username: str) -> str:
    """Create JWT token"""
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> str:
    """Verify JWT token and return username"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_current_user(request: Request) -> str:
    """Get current user from cookie"""
    token = request.cookies.get("access_token")
    if not token:
        return None
    return verify_token(token)

def user_exists(username: str) -> bool:
    """Check if user exists"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        return cursor.fetchone() is not None

def email_exists(email: str) -> bool:
    """Check if email exists"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        return cursor.fetchone() is not None

def create_user(username: str, email: str, password: str) -> bool:
    """Create new user"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            hashed_pwd = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, hashed_pwd)
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user:
            print(f"  ⚠ User '{username}' not found in database")
            return False
        is_valid = verify_password(password, user["password_hash"])
        if is_valid:
            print(f"  ✓ Password verified for '{username}'")
        else:
            print(f"  ✗ Password mismatch for '{username}'")
        return is_valid

# ============================================================================
# Initialize FastAPI
# ============================================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize database
init_db()

# ============================================================================
# Config
# ============================================================================
CHART_HISTORY_DAYS = 60

# ============================================================================
# Load Models and Scalers
# ============================================================================
lstm_model_adani = load_model("lstm_close_model.h5", compile=False)
scaler_adani = joblib.load("lstm_scaler.pkl")

try:
    lstm_model_bse = load_model("bse_model.h5", compile=False)
    scaler_bse = joblib.load("bse_scaler.pkl")
    print("✓ BSE model loaded successfully")
except:
    lstm_model_bse = lstm_model_adani
    scaler_bse = scaler_adani
    print("⚠ BSE model not found, using ADANIPORTS model")

try:
    lstm_model_nifty = load_model("nifty_model.h5", compile=False)
    scaler_nifty = joblib.load("nifty_scaler.pkl")
    print("✓ NIFTY model loaded successfully")
except:
    lstm_model_nifty = lstm_model_adani
    scaler_nifty = scaler_adani
    print("⚠ NIFTY model not found, using ADANIPORTS model")

# ============================================================================
# Load Historical Data
# ============================================================================
df_adani = pd.read_csv("ADANIPORTS.csv", parse_dates=["Date"])
df_adani = df_adani.sort_values("Date").reset_index(drop=True)

try:
    df_bse = pd.read_csv("BSE.csv")
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
    
    if column_mapping:
        df_bse = df_bse.rename(columns=column_mapping)
    df_bse['Date'] = pd.to_datetime(df_bse['Date'])
    df_bse = df_bse.sort_values("Date").reset_index(drop=True)
except:
    df_bse = df_adani.copy()

try:
    df_nifty = pd.read_csv("NIFTY50.csv")
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
    
    if column_mapping:
        df_nifty = df_nifty.rename(columns=column_mapping)
    df_nifty['Date'] = pd.to_datetime(df_nifty['Date'])
    df_nifty = df_nifty.sort_values("Date").reset_index(drop=True)
except:
    df_nifty = df_adani.copy()

# ============================================================================
# Forecast function
# ============================================================================
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

def get_historical_chart_data(df: pd.DataFrame, days_of_history: int):
    """Extracts last N days of data for Chart.js"""
    df_recent = df.tail(days_of_history).copy()
    historical_dates = df_recent['Date'].dt.strftime('%Y-%m-%d').tolist()
    historical_prices = df_recent['Close'].tolist()
    return historical_dates, historical_prices

# ============================================================================
# Authentication Routes
# ============================================================================
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    """Register page"""
    current_user = get_current_user(request)
    if current_user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    """Register new user"""
    error = None
    
    if len(username) < 3:
        error = "Username must be at least 3 characters long"
    elif len(password) < 6:
        error = "Password must be at least 6 characters long"
    elif password != confirm_password:
        error = "Passwords do not match"
    elif user_exists(username):
        error = "Username already taken"
    elif email_exists(email):
        error = "Email already registered"
    
    if error:
        return templates.TemplateResponse("register.html", {"request": request, "error": error})
    
    if create_user(username, email, password):
        response = RedirectResponse(url="/login?success=registered", status_code=302)
        return response
    
    return templates.TemplateResponse("register.html", {"request": request, "error": "Registration failed"})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    """Login page"""
    current_user = get_current_user(request)
    if current_user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Login user"""
    print(f"🔍 Login attempt - Username: {username}")
    
    if not authenticate_user(username, password):
        print(f"❌ Authentication failed for: {username}")
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    
    print(f"✅ Login successful for: {username}")
    token = create_access_token(username)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True, max_age=60*60*24*7)
    return response

@app.get("/logout")
def logout():
    """Logout user"""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="access_token")
    return response

# ============================================================================
# Protected Routes
# ============================================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page"""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    hist_dates, hist_prices = get_historical_chart_data(df_adani, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "username": current_user,
        "forecast": None,
        "stock_name": "ADANIPORTS (NSE)",
        "last_close": float(df_adani["Close"].iloc[-1]),
        "last_date": df_adani["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "adaniports",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices,
        "show_logout": True
    })

@app.post("/forecast", response_class=HTMLResponse)
def forecast(request: Request, days: int = Form(...)):
    """Forecast ADANIPORTS"""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    forecasted = forecast_next_days_lstm(df_adani, lstm_model_adani, scaler_adani, n_days=days)
    hist_dates, hist_prices = get_historical_chart_data(df_adani, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "username": current_user,
        "forecast": forecasted,
        "stock_name": "ADANIPORTS (NSE)",
        "days": days,
        "last_close": float(df_adani["Close"].iloc[-1]),
        "last_date": df_adani["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "adaniports",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

@app.get("/bse", response_class=HTMLResponse)
def bse_page(request: Request):
    """BSE page"""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    hist_dates, hist_prices = get_historical_chart_data(df_bse, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("BSE.html", {
        "request": request,
        "username": current_user,
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
    """Forecast BSE"""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    forecasted = forecast_next_days_lstm(df_bse, lstm_model_bse, scaler_bse, n_days=days)
    hist_dates, hist_prices = get_historical_chart_data(df_bse, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("BSE.html", {
        "request": request,
        "username": current_user,
        "forecast": forecasted,
        "stock_name": "BSE SENSEX",
        "days": days,
        "last_close": float(df_bse["Close"].iloc[-1]),
        "last_date": df_bse["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "bse",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

@app.get("/nifty", response_class=HTMLResponse)
def nifty_page(request: Request):
    """NIFTY page"""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    hist_dates, hist_prices = get_historical_chart_data(df_nifty, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("nifty.html", {
        "request": request,
        "username": current_user,
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
    """Forecast NIFTY"""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)
    
    forecasted = forecast_next_days_lstm(df_nifty, lstm_model_nifty, scaler_nifty, n_days=days)
    hist_dates, hist_prices = get_historical_chart_data(df_nifty, CHART_HISTORY_DAYS)
    
    return templates.TemplateResponse("nifty.html", {
        "request": request,
        "username": current_user,
        "forecast": forecasted,
        "stock_name": "NIFTY 50",
        "days": days,
        "last_close": float(df_nifty["Close"].iloc[-1]),
        "last_date": df_nifty["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_page": "nifty",
        "historical_dates": hist_dates,
        "historical_prices": hist_prices
    })

@app.get("/api/stocks")
def list_stocks(request: Request):
    """List all stocks"""
    current_user = get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "stocks": [
            {"name": "ADANIPORTS", "endpoint": "/", "model": "lstm_close_model.h5"},
            {"name": "BSE SENSEX", "endpoint": "/bse", "model": "bse_model.h5 (or fallback)"},
            {"name": "NIFTY 50", "endpoint": "/nifty", "model": "nifty_model.h5 (or fallback)"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)