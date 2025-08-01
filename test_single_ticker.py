#!/usr/bin/env python3
"""
Test script to verify the main functionality works with a single ticker.
This script tests the complete workflow without sending Telegram messages.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Activation, BatchNormalization, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import pickle
from keras.models import load_model

# Import our functions
from main import create_dataset, split_data, smape, mase, save_model_and_scaler, load_model_and_scaler, evalModel

def test_single_ticker(ticker='BTC-USD', test_period='1y'):
    """Test the complete workflow with a single ticker."""
    print(f"Testing with ticker: {ticker}")
    
    try:
        # Download the data
        print("Downloading data...")
        data = yf.download(ticker, period=test_period, interval='1d', timeout=20)
        
        if data.empty:
            print(f"No data downloaded for {ticker}")
            return False
            
        print(f"Downloaded {len(data)} days of data")
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create dataset
        X, y = create_dataset(scaled_data)
        X_train, y_train, X_test, y_test = split_data(X, y)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Try to load existing model first
        model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        if model is not None:
            print(f"Using saved model with MAE: {loaded_mae:.4f}")
            scaler = loaded_scaler
            mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
        else:
            print("Training new model...")
            # Build and train new model (simplified for testing)
            model = Sequential([
                Dense(50, input_shape=(X_train.shape[1], 1)),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error',
                          metrics=['mean_absolute_error', 'mean_squared_error'])
            
            # Train the model briefly for testing
            model.fit(X_train, y_train, epochs=5, batch_size=32,
                      validation_split=0.2, verbose=1)
            
            print(model.summary())
            
            mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
            
            # Save model if MAE is <= 10%
            if mae <= 0.10:  # 10% threshold
                save_model_and_scaler(model, scaler, ticker, mae)
                print(f"Model saved with MAE: {mae:.4f}")
            else:
                print(f"Model MAE {mae:.4f} > 10%, not saving model")
        
        # Test prediction
        print("Testing prediction...")
        data_recent = data.iloc[-60:]  # Get the last 60 days
        closing_prices = data_recent['Close'].values
        
        # Scale the recent data
        scaler_recent = MinMaxScaler(feature_range=(0, 1))
        scaled_recent = scaler_recent.fit_transform(closing_prices.reshape(-1, 1))
        
        # Predict next 5 days
        predicted_prices = []
        current_batch = scaled_recent[-60:].reshape(1, 60, 1)
        
        for i in range(5):  # Predicting 5 days
            next_prediction = model.predict(current_batch, verbose=0)
            next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
            current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
            predicted_prices.append(scaler_recent.inverse_transform(next_prediction)[0, 0])
        
        print(f"Predicted prices for next 5 days: {predicted_prices}")
        
        # Test file creation
        print("Testing file creation...")
        clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.plot(data.index[-30:], data['Close'][-30:], label='Actual Data')
        plt.title(f'{ticker} - Last 30 Days')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        # Save plot
        image_name = f'{clean_ticker}_test_predictions.png'
        plt.savefig(image_name)
        plt.close()
        
        # Verify file was created
        if os.path.exists(image_name):
            print(f"✅ Successfully created: {image_name}")
            os.remove(image_name)  # Clean up
        else:
            print(f"❌ Failed to create: {image_name}")
            return False
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    # Test with a simple ticker
    success = test_single_ticker('BTC-USD', '1y')
    
    if success:
        print("\n🎉 All functionality is working correctly!")
    else:
        print("\n💥 There are issues that need to be fixed.") 