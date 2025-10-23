import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

def train_model(csv_file, model_name, scaler_name):
    """
    Train LSTM model for stock price prediction
    Handles both BSE and NIFTY CSV formats automatically
    """
    print(f"\n{'='*60}")
    print(f"Training model for: {csv_file}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        return False
    
    # Load data - read first to detect format
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} rows from {csv_file}")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Detect and standardize column names
    # BSE format: Date,Open,High,Low,Close (UPPERCASE first letter)
    # NIFTY format: date,open,high,low,close,volume (lowercase)
    
    column_mapping = {}
    for col in df.columns:
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
    
    # Rename columns to standard format (only if mapping exists)
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"✓ Standardized columns: {df.columns.tolist()}")
    else:
        print(f"✓ Columns already in standard format: {df.columns.tolist()}")
    
    # Check if Close column exists
    if 'Close' not in df.columns:
        print(f"❌ Error: No 'Close' column found!")
        return False
    
    # Convert Date to datetime and sort
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"✓ Data sorted by date")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Prepare close prices
    close_prices = df['Close'].values.reshape(-1, 1)
    
    # Remove NaN values
    if np.isnan(close_prices).any():
        print("⚠ Warning: NaN values found, removing them...")
        valid_idx = ~np.isnan(close_prices).flatten()
        close_prices = close_prices[valid_idx]
        close_prices = close_prices.reshape(-1, 1)
    
    print(f"✓ Total data points: {len(close_prices)}")
    print(f"  Price range: ₹{close_prices.min():.2f} to ₹{close_prices.max():.2f}")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)
    print(f"✓ Data scaled to range [0, 1]")
    
    # Create sequences
    n_steps = 10
    X, y = [], []
    
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i, 0])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"✓ Created {len(X)} training sequences (using {n_steps} day lookback)")
    
    # Split into train and validation (80-20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    
    # Build LSTM model
    print(f"\n🔨 Building LSTM model...")
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\n🚀 Training started (10 epochs)...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        shuffle=False  # Don't shuffle time series data
    )
    
    # Evaluate
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"\n" + "="*60)
    print(f"✅ Training completed!")
    print(f"="*60)
    print(f"  Training Loss (MSE): {final_loss:.6f}")
    print(f"  Validation Loss (MSE): {final_val_loss:.6f}")
    print(f"  Training MAE: {final_mae:.6f}")
    print(f"  Validation MAE: {final_val_mae:.6f}")
    
    # Save model and scaler
    model.save(model_name)
    joblib.dump(scaler, scaler_name)
    
    print(f"\n💾 Files saved:")
    print(f"  ✓ Model: {model_name}")
    print(f"  ✓ Scaler: {scaler_name}")
    
    return True

# ============================================================================
# Main Training Script
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 LSTM MODEL TRAINING FOR INDIAN STOCK MARKET")
    print("="*60)
    
    models_to_train = [
        {
            "csv": "BSE.csv",
            "model": "bse_model.h5",
            "scaler": "bse_scaler.pkl",
            "name": "BSE SENSEX"
        },
        {
            "csv": "NIFTY50.csv",
            "model": "nifty_model.h5",
            "scaler": "nifty_scaler.pkl",
            "name": "NIFTY 50"
        }
    ]
    
    results = []
    
    for config in models_to_train:
        success = train_model(
            csv_file=config["csv"],
            model_name=config["model"],
            scaler_name=config["scaler"]
        )
        results.append({
            "name": config["name"],
            "csv": config["csv"],
            "success": success
        })
    
    # Final Summary
    print("\n\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    success_count = 0
    for result in results:
        if result["success"]:
            print(f"✅ {result['name']} ({result['csv']}): SUCCESS")
            success_count += 1
        else:
            print(f"❌ {result['name']} ({result['csv']}): FAILED")
    
    print("="*60)
    print(f"Models trained: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n🎉 All models trained successfully!")
        print("\n✓ You can now run your FastAPI app:")
        print("  python main.py")
        print("\n✓ Your app will use:")
        print("  • BSE predictions → bse_model.h5")
        print("  • NIFTY predictions → nifty_model.h5")
    else:
        print("\n⚠ Some models failed to train. Check the errors above.")
    
    print("="*60 + "\n")