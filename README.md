# StockTrendAI - LSTM Stock Market Forecasting Model

<img width="1883" height="873" alt="image" src="https://github.com/user-attachments/assets/18a5f2f9-f986-4284-af07-81dd1217caf6" />
<img width="1860" height="872" alt="image" src="https://github.com/user-attachments/assets/c566052e-55bc-4764-9a9b-819881e6710a" />
<img width="1913" height="890" alt="image" src="https://github.com/user-attachments/assets/53be1109-0d78-4780-998a-83533c54c0c3" />


An intelligent stock market forecasting application using LSTM neural networks to predict closing prices for Indian stock market indices. Track and forecast **ADANIPORTS (NSE)**, **BSE SENSEX**, and **NIFTY 50** with AI-powered predictions.

## 🎯 Features

- **Multi-Index Support**: Forecasts for ADANIPORTS, BSE SENSEX, and NIFTY 50
- **LSTM Neural Networks**: Deep learning models trained on historical stock data
- **Automatic Fallback**: Uses ADANIPORTS model as fallback if specific models aren't available
- **Historical Charts**: 60-day historical price visualization
- **Web Interface**: Clean, interactive UI built with FastAPI and Jinja2
- **Flexible Input**: Support for different CSV column formats (uppercase/lowercase handling)
- **RESTful API**: JSON endpoints for programmatic access

## 📋 Project Structure

```
share_market-situ/
├── main.py                    # FastAPI application server
├── train.py                   # Model training script
├── ADANIPORTS.csv            # ADANIPORTS historical data
├── BSE.csv                   # BSE SENSEX historical data
├── NIFTY50.csv               # NIFTY 50 historical data
├── lstm_close_model.h5       # ADANIPORTS LSTM model
├── lstm_scaler.pkl           # ADANIPORTS data scaler
├── bse_model.h5              # BSE-specific LSTM model
├── bse_scaler.pkl            # BSE data scaler
├── nifty_model.h5            # NIFTY-specific LSTM model
├── nifty_scaler.pkl          # NIFTY data scaler
├── static/
│   └── style.css             # Styling
├── templates/
│   ├── index.html            # ADANIPORTS page
│   ├── BSE.html              # BSE SENSEX page
│   └── nifty.html            # NIFTY 50 page
├── dataset/                  # Additional datasets
└── __pycache__/              # Python cache
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/situk0000/share_market-situ.git
   cd StockTrendAI
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn tensorflow keras jinja2 joblib
   ```

   Required packages:
   - FastAPI
   - Uvicorn
   - Pandas
   - NumPy
   - Scikit-learn
   - TensorFlow/Keras
   - Jinja2
   - Joblib

3. **Prepare CSV data files**
   - Ensure you have `ADANIPORTS.csv`, `BSE.csv`, and `NIFTY50.csv` in the project root
   - CSV format should include: Date, Open, High, Low, Close, Volume (column names are case-insensitive)

### Training Models

Train the LSTM models on your historical data:

```bash
python train.py
```

This will:
- Train BSE-specific model → `bse_model.h5` and `bse_scaler.pkl`
- Train NIFTY-specific model → `nifty_model.h5` and `nifty_scaler.pkl`
- Use ADANIPORTS model as fallback if specific models aren't available
- Display training metrics and save all models automatically

### Running the Application

Start the FastAPI server:

```bash
python main.py
```

The application will be available at: `http://localhost:8000`

Access different stock indices:
- **ADANIPORTS**: http://localhost:8000/
- **BSE SENSEX**: http://localhost:8000/bse
- **NIFTY 50**: http://localhost:8000/nifty

## 📊 How It Works

### Training Process

1. **Data Preprocessing**: Normalize prices using MinMaxScaler (0-1 range)
2. **Sequence Creation**: Create 10-day lookback sequences for training
3. **Train/Validation Split**: 80% training, 20% validation
4. **Optimization**: Adam optimizer with MSE loss function
5. **Regularization**: Dropout layers to prevent overfitting

### Forecasting

Given historical data, the model:
1. Takes the last 10 days of prices
2. Predicts the next day's closing price
3. Adds prediction to sequence and removes oldest value
4. Repeats for the requested number of days

## 🔌 API Endpoints

### GET `/`
Home page - ADANIPORTS forecast interface

### POST `/forecast`
Generate forecast for ADANIPORTS
- **Parameters**: `days` (Form) - number of days to forecast (1-30)
- **Response**: HTML page with forecast chart

### GET `/bse`
BSE SENSEX forecast page

### POST `/forecast_bse`
Generate forecast for BSE SENSEX
- **Parameters**: `days` (Form) - number of days to forecast

### GET `/nifty`
NIFTY 50 forecast page

### POST `/forecast_nifty`
Generate forecast for NIFTY 50
- **Parameters**: `days` (Form) - number of days to forecast

### GET `/api/stocks`
List available stocks and their endpoints (JSON)

```json
{
  "stocks": [
    {"name": "ADANIPORTS", "endpoint": "/", "model": "lstm_close_model.h5"},
    {"name": "BSE SENSEX", "endpoint": "/bse", "model": "bse_model.h5"},
    {"name": "NIFTY 50", "endpoint": "/nifty", "model": "nifty_model.h5"}
  ],
  "note": "Each index uses its own model if available, otherwise falls back to ADANIPORTS model"
}
```

## 📈 Data Format

CSV files should include these columns (case-insensitive):
- **Date**: Trading date (YYYY-MM-DD format)
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price (used for predictions)
- **Volume**: Trading volume


## 📝 Model Performance

The model is trained with:
- **Loss Function**: Mean Squared Error (MSE)
- **Metric**: Mean Absolute Error (MAE)
- **Optimizer**: Adam
- **Epochs**: 10

Training logs show:
- Training loss and validation loss
- Mean Absolute Error metrics
- Final model accuracy

## 🛠️ Troubleshooting

### Model not found warning
If you see `"⚠ BSE model not found, using ADANIPORTS model"`:
- Run `python train.py` to train the missing models
- Ensure CSV files exist in the project root

### CSV parsing errors
- Verify column names (case doesn't matter, but should be: Date, Open, High, Low, Close, Volume)
- Check for missing or malformed data
- Ensure dates are in YYYY-MM-DD format

### Port already in use
Change the port in `main.py`:
```bash
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use 8001 instead of 8000
```

## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors before making investment decisions.

## 📄 License

This project is licensed under the MIT License 

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Built with FastAPI and TensorFlow/Keras
- Historical data from Indian stock markets

---

## Author 
Situ Kumari - situk0000
