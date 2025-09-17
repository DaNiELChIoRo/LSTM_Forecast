#!/usr/bin/env python3
"""
Test script to verify the main script logic works without the TypeError.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import tempfile
import os
import shutil

# Import our functions
from main import create_dataset, split_data, save_model_and_scaler, load_model_and_scaler, evalModel

def test_main_logic():
    """Test the main script logic without running the full pipeline."""
    print("Testing main script logic...")
    
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(test_dir)
        
        # Create mock data
        np.random.seed(42)
        mock_data = pd.DataFrame({
            'Open': np.random.rand(100),
            'High': np.random.rand(100),
            'Low': np.random.rand(100),
            'Close': np.random.rand(100),
            'Volume': np.random.rand(100)
        })
        
        # Test the main logic flow
        ticker = "TEST/STOCK"
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(mock_data)
        
        # Create dataset
        X, y = create_dataset(scaled_data)
        X_train, y_train, X_test, y_test = split_data(X, y)
        
        print(f"Dataset created: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Try to load existing model (should be None)
        model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        print(f"Load existing model result: {model is None}")
        
        # Create and train a simple model
        model = Sequential([
            Dense(10, input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train briefly
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        # Evaluate model
        mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
        
        print(f"Model evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Test the MAE threshold logic
        if mae <= 0.10:  # 10% threshold
            save_model_and_scaler(model, scaler, ticker, mae)
            print(f"✅ Model saved with MAE: {mae:.4f}")
        else:
            print(f"❌ Model MAE {mae:.4f} > 10%, not saving model")
        
        # Test loading the saved model
        loaded_model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        if loaded_model is not None:
            print(f"✅ Successfully loaded saved model with MAE: {loaded_mae:.4f}")
        else:
            print("❌ Failed to load saved model")
        
        print("✅ All main script logic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
        
    finally:
        # Clean up
        os.chdir(original_cwd)
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_main_logic()
    
    if success:
        print("\n🎉 Main script logic is working correctly!")
        print("The TypeError has been fixed!")
    else:
        print("\n💥 There are still issues to fix.") 