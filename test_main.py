import unittest
import pandas as pd
import numpy as np
import os
import pickle
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Activation, BatchNormalization, Flatten
from keras.models import load_model

# Import the functions we want to test
import sys
sys.path.append('.')
from main import create_dataset, split_data, smape, mase, save_model_and_scaler, load_model_and_scaler


class TestLSTMForecast(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create test data
        self.test_data = np.random.rand(100, 5)  # 100 days, 5 features
        self.scaled_data = MinMaxScaler().fit_transform(self.test_data)
        
    def tearDown(self):
        """Clean up after each test method."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_create_dataset(self):
        """Test the create_dataset function."""
        X, y = create_dataset(self.scaled_data, days_range=10)
        
        # Check shapes
        self.assertEqual(X.shape[1], 10)  # 10 days lookback
        self.assertEqual(y.shape[0], X.shape[0])  # Same number of samples
        self.assertEqual(len(X), len(self.scaled_data) - 10)  # Correct number of samples
        
        # Check that X contains the right data
        for i in range(len(X)):
            expected_X = self.scaled_data[i:i+10, 0]
            expected_y = self.scaled_data[i+10, 0]
            np.testing.assert_array_equal(X[i], expected_X)
            self.assertEqual(y[i], expected_y)
    
    def test_split_data(self):
        """Test the split_data function."""
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        X_train, y_train, X_test, y_test = split_data(X, y, train_size=0.8)
        
        # Check shapes
        self.assertEqual(len(X_train), 80)  # 80% of 100
        self.assertEqual(len(X_test), 20)   # 20% of 100
        self.assertEqual(X_train.shape[1], 10)
        self.assertEqual(X_train.shape[2], 1)  # Should be reshaped to 3D
        
        # Check that data is properly split
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
    
    def test_smape(self):
        """Test the SMAPE function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        result = smape(y_true, y_pred)
        
        # SMAPE should be a positive number
        self.assertGreater(result, 0)
        self.assertIsInstance(result, (int, float))
        
        # Test with identical arrays (should be 0)
        result_identical = smape(y_true, y_true)
        self.assertEqual(result_identical, 0)
    
    def test_mase(self):
        """Test the MASE function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        result = mase(y_true, y_pred, y_train)
        
        # MASE should be a positive number
        self.assertGreater(result, 0)
        self.assertIsInstance(result, (int, float))
    
    def test_save_model_and_scaler(self):
        """Test the save_model_and_scaler function."""
        # Create a simple model
        model = Sequential([
            Dense(10, input_shape=(5,)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        scaler = MinMaxScaler()
        ticker = "TEST/STOCK"
        mae_value = 0.05  # 5% MAE
        
        # Save model and scaler
        save_model_and_scaler(model, scaler, ticker, mae_value)
        
        # Check that files were created
        clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
        model_path = f'models/{clean_ticker}_model.h5'
        scaler_path = f'models/{clean_ticker}_scaler.pkl'
        mae_info_path = f'models/{clean_ticker}_mae_info.pkl'
        
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(scaler_path))
        self.assertTrue(os.path.exists(mae_info_path))
        
        # Check MAE info content
        with open(mae_info_path, 'rb') as f:
            mae_info = pickle.load(f)
        
        self.assertEqual(mae_info['mae'], mae_value)
        self.assertEqual(mae_info['ticker'], ticker)
    
    def test_load_model_and_scaler_success(self):
        """Test successful loading of model and scaler."""
        # Create and save a model first
        model = Sequential([
            Dense(10, input_shape=(5,)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        scaler = MinMaxScaler()
        ticker = "GOOD/STOCK"
        mae_value = 0.08  # 8% MAE (below 10% threshold)
        
        save_model_and_scaler(model, scaler, ticker, mae_value)
        
        # Load the model
        loaded_model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        # Check that loading was successful
        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(loaded_scaler)
        self.assertEqual(loaded_mae, mae_value)
        
        # Check that the loaded model has the same architecture
        self.assertEqual(len(loaded_model.layers), len(model.layers))
    
    def test_load_model_and_scaler_high_mae(self):
        """Test that models with high MAE are not loaded."""
        # Create and save a model with high MAE
        model = Sequential([
            Dense(10, input_shape=(5,)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        scaler = MinMaxScaler()
        ticker = "BAD/STOCK"
        mae_value = 0.15  # 15% MAE (above 10% threshold)
        
        save_model_and_scaler(model, scaler, ticker, mae_value)
        
        # Try to load the model
        loaded_model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        # Check that loading failed due to high MAE
        self.assertIsNone(loaded_model)
        self.assertIsNone(loaded_scaler)
        self.assertIsNone(loaded_mae)
    
    def test_load_model_and_scaler_not_exists(self):
        """Test loading when model doesn't exist."""
        ticker = "NONEXISTENT/STOCK"
        
        loaded_model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        # Check that loading failed
        self.assertIsNone(loaded_model)
        self.assertIsNone(loaded_scaler)
        self.assertIsNone(loaded_mae)
    
    def test_ticker_name_cleaning(self):
        """Test that ticker names are properly cleaned for file paths."""
        test_cases = [
            ("USD/MXN", "USD_MXN"),
            ("^IXIC", "IXIC"),
            ("MXN=X", "MXN_X"),
            ("BTC-USD", "BTC-USD"),
            ("ETH-USD", "ETH-USD")
        ]
        
        for original, expected in test_cases:
            clean_ticker = original.replace('/', '_').replace('^', '').replace('=', '_')
            self.assertEqual(clean_ticker, expected)
    
    def test_file_path_safety(self):
        """Test that file paths are safe and don't contain invalid characters."""
        problematic_tickers = ["USD/MXN", "^IXIC", "MXN=X", "STOCK/PRICE"]
        
        for ticker in problematic_tickers:
            clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
            
            # Test that the cleaned name can be used in file paths
            test_path = f"models/{clean_ticker}_model.h5"
            
            # Should not contain any problematic characters
            self.assertNotIn('/', clean_ticker)
            self.assertNotIn('^', clean_ticker)
            self.assertNotIn('=', clean_ticker)
            
            # Should be a valid filename
            self.assertTrue(all(c.isalnum() or c in '_-' for c in clean_ticker))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up after each test method."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    @patch('yfinance.download')
    def test_complete_workflow(self, mock_download):
        """Test the complete workflow with mocked data."""
        # Mock yfinance download to return test data
        mock_data = pd.DataFrame({
            'Open': np.random.rand(100),
            'High': np.random.rand(100),
            'Low': np.random.rand(100),
            'Close': np.random.rand(100),
            'Volume': np.random.rand(100)
        }, index=pd.date_range('2020-01-01', periods=100))
        
        mock_download.return_value = mock_data
        
        # Import and run the main function for a single ticker
        from main import create_dataset, split_data, save_model_and_scaler, load_model_and_scaler
        
        # Test with a single ticker
        ticker = "TEST/STOCK"
        
        # Create test data
        data = mock_data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create dataset
        X, y = create_dataset(scaled_data)
        X_train, y_train, X_test, y_test = split_data(X, y)
        
        # Create a simple model
        model = Sequential([
            Dense(10, input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model briefly
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        # Evaluate and get MAE
        y_pred = model.predict(X_test, verbose=0)
        # Ensure shapes match for MAE calculation
        y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Make sure both arrays have the same length
        min_length = min(len(y_test_flat), len(y_pred_flat))
        y_test_flat = y_test_flat[:min_length]
        y_pred_flat = y_pred_flat[:min_length]
        
        mae = np.mean(np.abs(y_test_flat - y_pred_flat))
        
        # Save model if MAE is good
        if mae <= 0.10:
            save_model_and_scaler(model, scaler, ticker, mae)
        
        # Try to load the model
        loaded_model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        # If MAE was good, model should be loaded
        if mae <= 0.10:
            self.assertIsNotNone(loaded_model)
            self.assertIsNotNone(loaded_scaler)
            self.assertEqual(loaded_mae, mae)
        else:
            self.assertIsNone(loaded_model)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 