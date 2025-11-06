# 📊 StockTrendAI - Stock Market Forecasting Model (LSTM)

An AI-powered stock market prediction application using LSTM neural networks with user authentication. This application forecasts closing prices for Indian stock market indices (ADANIPORTS, BSE SENSEX, NIFTY 50) using historical data and deep learning.
<img width="1885" height="895" alt="image" src="https://github.com/user-attachments/assets/59280a37-2029-44da-b76f-7574bb4f16a0" />

## ✨ Features

- **User Authentication**: Secure login and registration system
- **JWT Token-based Sessions**: HTTP-only cookies for session management
- **Password Security**: Argon2 hashing for secure password storage
- **LSTM Deep Learning Model**: Predicts stock prices for the next N days
- **Multi-Index Support**: 
  - ADANIPORTS (NSE)
  - BSE SENSEX
  - NIFTY 50
- **Historical Charts**: Visualize last 60 days of historical data
- **Real-time Predictions**: Generate forecasts for 1-30 days ahead
- **Responsive UI**: Clean, modern interface with gradient design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/situk0000/share_market-situ.git
cd share_market-situ
```
2. **Install argon2 for password hashing**
```bash
pip install argon2-cffi
```

3. **Run the application**
```bash
uvicorn main:app --reload
```

4. **Open in browser**
```
http://localhost:8000
```

## 📋 Requirements

- fastapi
- uvicorn
- pandas
- numpy
- tensorflow
- scikit-learn
- joblib
- passlib
- PyJWT
- argon2-cffi
- Jinja2

## 🏗️ Project Structure

```
share_market-situ/
├── main.py                 # FastAPI application with authentication
├── train.py               # LSTM model training script
├── templates/
│   ├── index.html        # ADANIPORTS forecast page
│   ├── BSE.html          # BSE SENSEX forecast page
│   ├── nifty.html        # NIFTY 50 forecast page
│   ├── login.html        # Login page
│   └── register.html     # Registration page
├── static/
│   └── style.css         # Styling
├── ADANIPORTS.csv        # Historical data for ADANIPORTS
├── BSE.csv              # Historical data for BSE
├── NIFTY50.csv          # Historical data for NIFTY 50
├── lstm_close_model.h5  # Pre-trained LSTM model (ADANIPORTS)
├── bse_model.h5         # Pre-trained LSTM model (BSE)
├── nifty_model.h5       # Pre-trained LSTM model (NIFTY 50)
└── users.db             # SQLite database (auto-created)
```

## 🔐 Authentication System

### Register
1. Navigate to `http://localhost:8000/register`
2. Enter username (min 3 characters)
3. Enter email
4. Enter password (min 6 characters)
5. Confirm password
6. Click "Register"

### Login
1. Navigate to `http://localhost:8000/login`
2. Enter your username
3. Enter your password
4. Click "Login"

### Logout
Click the "Logout" button in the top-right corner of any forecast page.

## 🤖 How the LSTM Model Works

The application uses **Long Short-Term Memory (LSTM)** neural networks to predict stock prices:

1. **Data Preparation**: Historical stock prices are normalized using MinMaxScaler
2. **Sequence Creation**: Creates sequences of 10 days to predict the next day's close price
3. **Model Architecture**:
   - 2 LSTM layers (50 units each) with Dropout
   - Dense layers for output prediction
   - Adam optimizer with MSE loss
4. **Training**: Trained on 80% of historical data, validated on 20%
5. **Forecasting**: Uses the last 10 days to predict future prices

## 📊 API Endpoints

### Authentication Routes
- `GET /login` - Display login page
- `POST /login` - Submit login credentials
- `GET /register` - Display registration page
- `POST /register` - Submit registration form
- `GET /logout` - Logout user

### Stock Forecast Routes (Protected)
- `GET /` - ADANIPORTS forecast page
- `POST /forecast` - Generate ADANIPORTS forecast
- `GET /bse` - BSE SENSEX forecast page
- `POST /forecast_bse` - Generate BSE forecast
- `GET /nifty` - NIFTY 50 forecast page
- `POST /forecast_nifty` - Generate NIFTY forecast

### API Endpoints
- `GET /api/stocks` - List available stocks

## 🎯 Usage Example

1. **Register/Login** to access the application
2. **Select a stock index** (ADANIPORTS, BSE, or NIFTY)
3. **Enter number of days** to forecast (1-30)
4. **Click "Generate Forecast"**
5. **View predictions** in the results table
6. **Check historical charts** to understand trends

## 🔒 Security Features

- ✅ Password hashing with Argon2
- ✅ JWT token-based authentication
- ✅ HTTP-only cookies (prevents XSS attacks)
- ✅ Session expiration (7 days)
- ✅ SQLite database for user storage
- ✅ Protected routes requiring authentication
- ✅ Unique username and email constraints

## 🧠 Training the Models

To retrain the LSTM models with new data:

```bash
python train.py
```

This will:
1. Load CSV data from ADANIPORTS.csv, BSE.csv, and NIFTY50.csv
2. Prepare and normalize the data
3. Create training sequences
4. Train the LSTM models
5. Save models and scalers

## 📈 Expected Output

The application generates predictions like:

| Date | Predicted Close Price (₹) |
|------|--------------------------|
| 2025-11-07 | ₹10,850.50 |
| 2025-11-08 | ₹10,920.75 |
| 2025-11-09 | ₹10,995.00 |

## ⚙️ Configuration

**Authentication Settings** (in `main.py`):
```python
SECRET_KEY = "Gojo Saturo"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7
```

**Forecast Settings** (in `main.py`):
```python
CHART_HISTORY_DAYS = 60  # Days to show in historical chart
```

## 🐛 Troubleshooting

### "Template not found" error
```bash
# Ensure templates folder exists with all HTML files
mkdir -p templates
```

### Database errors
```bash
# Delete and recreate the database
rm users.db
uvicorn main:app --reload
```

### Argon2 errors
```bash
pip install argon2-cffi
```

## 📝 Notes

- The LSTM model uses the last 10 days of data to predict the next day
- Historical data should be in CSV format with columns: Date, Open, High, Low, Close, Volume
- Predictions are based on historical patterns and should not be used as financial advice
- Token expiration is set to 7 days; adjust `ACCESS_TOKEN_EXPIRE_DAYS` as needed

## 🔄 Future Improvements

- [ ] Email verification for registration
- [ ] Password reset functionality
- [ ] User profile management
- [ ] Advanced ML models (Prophet, XGBoost)
- [ ] Real-time data integration
- [ ] Mobile app version
- [ ] Portfolio tracking
- [ ] Risk analysis tools

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Situ Kumari**
- GitHub: [@situk0000](https://github.com/situk0000)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
