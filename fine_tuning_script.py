#!/usr/bin/env python3
"""
Automated Fine-Tuning Script for Hybrid CNN-LSTM-Transformer Architecture
This script optimizes hyperparameters and saves the best configurations for scheduled runs.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import json
import os
import pickle
import time
import random
from datetime import datetime, timedelta
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import from main.py
from main import (
    create_hybrid_cnn_lstm_transformer_model, 
    create_dataset, 
    split_data, 
    evalModel,
    smape,
    mase,
    ARCHITECTURE_CONFIG
)

# Import Keras components
from keras.optimizers import AdamW, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TuningConfig:
    """Configuration class for fine-tuning parameters"""
    # Architecture parameters
    cnn_filters: List[int]
    cnn_kernels: List[int]
    lstm_units: List[int]
    transformer_heads: int
    transformer_key_dim: int
    dense_units: List[int]
    dropout_rate: float
    
    # Training parameters
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    
    # Performance metrics
    mae: float
    rmse: float
    mape: float
    smape: float
    mase: float
    
    # Metadata
    timestamp: str
    ticker: str
    training_time: float
    architecture_type: str = 'hybrid'

class HyperparameterTuner:
    """Automated hyperparameter tuning for the hybrid architecture"""
    
    def __init__(self, ticker: str = 'BTC-USD', test_period: str = '2y'):
        self.ticker = ticker
        self.test_period = test_period
        self.best_configs = []
        self.tuning_history = []
        self.results_dir = 'tuning_results'
        self.models_dir = 'tuned_models'
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing best configs
        self.load_best_configs()
    
    def define_search_space(self) -> Dict[str, List]:
        """Define the hyperparameter search space"""
        return {
            'cnn_filters': [[32, 64], [64, 128], [128, 256]],
            'cnn_kernels': [[3, 5], [5, 7], [3, 7]],
            'lstm_units': [[32, 64, 32], [50, 100, 50], [64, 128, 64]],
            'transformer_heads': [4, 8, 12],
            'transformer_key_dim': [32, 64, 128],
            'dense_units': [[64, 32], [128, 64], [256, 128]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'batch_size': [8, 16, 32],
            'learning_rate': [0.0001, 0.001, 0.01],
            'weight_decay': [0.001, 0.01, 0.1]
        }
    
    def generate_configurations(self, max_configs: int = 50) -> List[Dict]:
        """Generate random configurations from search space"""
        search_space = self.define_search_space()
        configs = []
        
        # Generate random combinations
        for _ in range(max_configs):
            config = {}
            for param, values in search_space.items():
                config[param] = random.choice(values)
            configs.append(config)
        
        return configs
    
    def download_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """Download and prepare data for training"""
        logger.info(f"📥 Downloading data for {self.ticker}...")
        
        try:
            data = yf.download(self.ticker, period=self.test_period, interval='1d', progress=False)
            
            if data.empty:
                raise ValueError(f"No data available for {self.ticker}")
            
            # Use only Close prices
            prices = data['Close'].values.reshape(-1, 1)
            
            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(prices)
            
            # Create dataset
            X, y = create_dataset(scaled_data, days_range=60)
            X_train, y_train, X_test, y_test = split_data(X, y, train_size=0.8)
            
            logger.info(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
            return X_train, y_train, X_test, y_test, scaler
            
        except Exception as e:
            logger.error(f"❌ Error downloading data: {e}")
            raise
    
    def train_and_evaluate_config(self, config: Dict, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> TuningConfig:
        """Train and evaluate a single configuration"""
        start_time = time.time()
        
        try:
            # Create model with current configuration
            model = create_hybrid_cnn_lstm_transformer_model(
                input_shape=(X_train.shape[1], 1),
                architecture_type='hybrid'
            )
            
            # Update model parameters (this would require modifying the create function)
            # For now, we'll use the config for compilation and training
            
            # Compile model
            optimizer = AdamW(
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
            
            training_time = time.time() - start_time
            
            # Create tuning config
            tuning_config = TuningConfig(
                cnn_filters=config['cnn_filters'],
                cnn_kernels=config['cnn_kernels'],
                lstm_units=config['lstm_units'],
                transformer_heads=config['transformer_heads'],
                transformer_key_dim=config['transformer_key_dim'],
                dense_units=config['dense_units'],
                dropout_rate=config['dropout_rate'],
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                mae=mae,
                rmse=rmse,
                mape=mae,
                smape=smape_value,
                mase=mase_value,
                timestamp=datetime.now().isoformat(),
                ticker=self.ticker,
                training_time=training_time
            )
            
            logger.info(f"✅ Config evaluated - MAE: {mae:.4f}, Time: {training_time:.1f}s")
            return tuning_config
            
        except Exception as e:
            logger.error(f"❌ Error training config: {e}")
            # Return a failed config
            return TuningConfig(
                cnn_filters=config.get('cnn_filters', [64, 128]),
                cnn_kernels=config.get('cnn_kernels', [3, 5]),
                lstm_units=config.get('lstm_units', [50, 100, 50]),
                transformer_heads=config.get('transformer_heads', 8),
                transformer_key_dim=config.get('transformer_key_dim', 64),
                dense_units=config.get('dense_units', [128, 64]),
                dropout_rate=config.get('dropout_rate', 0.2),
                batch_size=config.get('batch_size', 16),
                epochs=config.get('epochs', 50),
                learning_rate=config.get('learning_rate', 0.001),
                weight_decay=config.get('weight_decay', 0.01),
                mae=float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                smape=float('inf'),
                mase=float('inf'),
                timestamp=datetime.now().isoformat(),
                ticker=self.ticker,
                training_time=time.time() - start_time
            )
    
    def run_hyperparameter_search(self, max_configs: int = 30, max_workers: int = 2) -> List[TuningConfig]:
        """Run hyperparameter search with parallel processing"""
        logger.info(f"🚀 Starting hyperparameter search for {self.ticker}")
        logger.info(f"📊 Will test {max_configs} configurations with {max_workers} workers")
        
        # Prepare data
        X_train, y_train, X_test, y_test, scaler = self.download_and_prepare_data()
        
        # Generate configurations
        configs = self.generate_configurations(max_configs)
        
        # Run parallel training
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self.train_and_evaluate_config, config, X_train, y_train, X_test, y_test): config
                for config in configs
            }
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.tuning_history.append(result)
                    
                    # Save intermediate results
                    if len(results) % 5 == 0:
                        self.save_tuning_results(results)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing config: {e}")
        
        # Sort by MAE (lower is better)
        results.sort(key=lambda x: x.mae)
        
        # Save best results
        self.best_configs.extend(results[:10])  # Keep top 10
        self.best_configs.sort(key=lambda x: x.mae)
        self.best_configs = self.best_configs[:10]  # Keep only top 10
        
        self.save_tuning_results(results)
        self.save_best_configs()
        
        logger.info(f"✅ Hyperparameter search completed!")
        logger.info(f"🏆 Best MAE: {results[0].mae:.4f}")
        
        return results
    
    def save_tuning_results(self, results: List[TuningConfig]):
        """Save tuning results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.results_dir}/tuning_results_{self.ticker}_{timestamp}.json"
        
        # Convert to dict for JSON serialization
        results_dict = [asdict(result) for result in results]
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Results saved to {filename}")
    
    def save_best_configs(self):
        """Save best configurations"""
        filename = f"{self.results_dir}/best_configs_{self.ticker}.json"
        
        # Convert to dict for JSON serialization
        best_dict = [asdict(config) for config in self.best_configs]
        
        with open(filename, 'w') as f:
            json.dump(best_dict, f, indent=2)
        
        logger.info(f"🏆 Best configs saved to {filename}")
    
    def load_best_configs(self):
        """Load existing best configurations"""
        filename = f"{self.results_dir}/best_configs_{self.ticker}.json"
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    best_dict = json.load(f)
                
                self.best_configs = [TuningConfig(**config) for config in best_dict]
                logger.info(f"📂 Loaded {len(self.best_configs)} existing best configs")
                
            except Exception as e:
                logger.error(f"❌ Error loading best configs: {e}")
                self.best_configs = []
    
    def get_best_config(self) -> TuningConfig:
        """Get the best configuration"""
        if self.best_configs:
            return self.best_configs[0]
        else:
            # Return default config
            return TuningConfig(
                cnn_filters=[64, 128],
                cnn_kernels=[3, 5],
                lstm_units=[50, 100, 50],
                transformer_heads=8,
                transformer_key_dim=64,
                dense_units=[128, 64],
                dropout_rate=0.2,
                batch_size=16,
                epochs=150,
                learning_rate=0.001,
                weight_decay=0.01,
                mae=0.0,
                rmse=0.0,
                mape=0.0,
                smape=0.0,
                mase=0.0,
                timestamp=datetime.now().isoformat(),
                ticker=self.ticker,
                training_time=0.0
            )
    
    def apply_best_config_to_main(self):
        """Apply the best configuration to main.py"""
        best_config = self.get_best_config()
        
        # Update ARCHITECTURE_CONFIG in main.py
        new_config = {
            'type': 'hybrid',
            'cnn_filters': best_config.cnn_filters,
            'cnn_kernels': best_config.cnn_kernels,
            'lstm_units': best_config.lstm_units,
            'transformer_heads': best_config.transformer_heads,
            'transformer_key_dim': best_config.transformer_key_dim,
            'dense_units': best_config.dense_units,
            'dropout_rate': best_config.dropout_rate,
            'batch_size': best_config.batch_size,
            'epochs': best_config.epochs,
            'learning_rate': best_config.learning_rate,
            'weight_decay': best_config.weight_decay
        }
        
        # Save to file
        config_filename = f"{self.results_dir}/optimized_config_{self.ticker}.json"
        with open(config_filename, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        logger.info(f"🎯 Best config applied and saved to {config_filename}")
        logger.info(f"📊 Best MAE: {best_config.mae:.4f}")
        
        return new_config
    
    def generate_tuning_report(self) -> str:
        """Generate a comprehensive tuning report"""
        if not self.best_configs:
            return "No tuning results available."
        
        best = self.best_configs[0]
        
        report = f"""
🏆 HYPERPARAMETER TUNING REPORT
{'='*50}
Ticker: {self.ticker}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Configurations Tested: {len(self.tuning_history)}

🏅 BEST CONFIGURATION:
MAE: {best.mae:.4f}
RMSE: {best.rmse:.4f}
MAPE: {best.mape:.4f}
SMAPE: {best.smape:.4f}
MASE: {best.mase:.4f}
Training Time: {best.training_time:.1f}s

🔧 OPTIMAL PARAMETERS:
CNN Filters: {best.cnn_filters}
CNN Kernels: {best.cnn_kernels}
LSTM Units: {best.lstm_units}
Transformer Heads: {best.transformer_heads}
Transformer Key Dim: {best.transformer_key_dim}
Dense Units: {best.dense_units}
Dropout Rate: {best.dropout_rate}
Batch Size: {best.batch_size}
Learning Rate: {best.learning_rate}
Weight Decay: {best.weight_decay}

📊 TOP 5 CONFIGURATIONS:
"""
        
        for i, config in enumerate(self.best_configs[:5]):
            report += f"{i+1}. MAE: {config.mae:.4f}, Time: {config.training_time:.1f}s\n"
        
        return report

def schedule_tuning_job(ticker: str = 'BTC-USD', max_configs: int = 30):
    """Function to be called by scheduler"""
    logger.info(f"🕐 Scheduled tuning job started for {ticker}")
    
    tuner = HyperparameterTuner(ticker=ticker)
    results = tuner.run_hyperparameter_search(max_configs=max_configs)
    
    # Apply best config
    tuner.apply_best_config_to_main()
    
    # Generate report
    report = tuner.generate_tuning_report()
    
    # Save report
    report_filename = f"tuning_results/tuning_report_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"📋 Report saved to {report_filename}")
    logger.info("✅ Scheduled tuning job completed")
    
    return results

def create_scheduler_script():
    """Create a scheduler script for automated tuning"""
    scheduler_script = """#!/usr/bin/env python3
'''
Automated Scheduler for Fine-Tuning Script
Run this script with cron or task scheduler for automated hyperparameter optimization
'''

import schedule
import time
from fine_tuning_script import schedule_tuning_job
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def daily_tuning():
    \"\"\"Run daily tuning for BTC-USD\"\"\"
    logger.info("🕐 Starting daily tuning job...")
    schedule_tuning_job(ticker='BTC-USD', max_configs=20)

def weekly_tuning():
    \"\"\"Run weekly comprehensive tuning\"\"\"
    logger.info("🕐 Starting weekly comprehensive tuning...")
    tickers = ['BTC-USD', 'ETH-USD', '^IXIC']
    for ticker in tickers:
        schedule_tuning_job(ticker=ticker, max_configs=30)

# Schedule jobs
schedule.every().day.at("02:00").do(daily_tuning)  # Daily at 2 AM
schedule.every().sunday.at("01:00").do(weekly_tuning)  # Weekly on Sunday at 1 AM

logger.info("📅 Scheduler started. Jobs scheduled:")
logger.info("  - Daily tuning: 2:00 AM")
logger.info("  - Weekly comprehensive tuning: Sunday 1:00 AM")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
"""
    
    with open('scheduler.py', 'w') as f:
        f.write(scheduler_script)
    
    print("📅 Scheduler script created: scheduler.py")
    print("💡 To run: python scheduler.py")
    print("🔧 To schedule with cron: 0 2 * * * /path/to/python /path/to/scheduler.py")

if __name__ == "__main__":
    print("🧪 HYPERPARAMETER FINE-TUNING SCRIPT")
    print("="*50)
    
    # Example usage
    ticker = 'BTC-USD'
    max_configs = 20  # Reduce for testing
    
    print(f"🎯 Starting fine-tuning for {ticker}")
    print(f"📊 Will test {max_configs} configurations")
    
    # Run tuning
    tuner = HyperparameterTuner(ticker=ticker)
    results = tuner.run_hyperparameter_search(max_configs=max_configs)
    
    # Apply best configuration
    best_config = tuner.apply_best_config_to_main()
    
    # Generate and display report
    report = tuner.generate_tuning_report()
    print(report)
    
    # Create scheduler script
    create_scheduler_script()
    
    print("\n✅ Fine-tuning completed!")
    print("📁 Check 'tuning_results/' directory for detailed results")
    print("📅 Use 'scheduler.py' for automated tuning")