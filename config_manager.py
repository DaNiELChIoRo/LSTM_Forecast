#!/usr/bin/env python3
"""
Configuration Manager for Automated Model Optimization
Handles loading and applying optimized configurations from fine-tuning results
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizedConfig:
    """Optimized configuration from fine-tuning"""
    ticker: str
    mae: float
    config: Dict
    timestamp: str
    training_time: float

class ConfigManager:
    """Manages optimized configurations from fine-tuning"""
    
    def __init__(self, results_dir: str = 'tuning_results'):
        self.results_dir = results_dir
        self.optimized_configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all optimized configurations"""
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory {self.results_dir} does not exist")
            return
        
        for filename in os.listdir(self.results_dir):
            if filename.startswith('optimized_config_') and filename.endswith('.json'):
                ticker = filename.replace('optimized_config_', '').replace('.json', '')
                self.load_config_for_ticker(ticker)
    
    def load_config_for_ticker(self, ticker: str) -> Optional[OptimizedConfig]:
        """Load optimized configuration for a specific ticker"""
        config_file = os.path.join(self.results_dir, f'optimized_config_{ticker}.json')
        
        if not os.path.exists(config_file):
            logger.warning(f"No optimized config found for {ticker}")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Extract metadata
            mae = config_data.get('mae', 0.0)
            timestamp = config_data.get('timestamp', datetime.now().isoformat())
            training_time = config_data.get('training_time', 0.0)
            
            optimized_config = OptimizedConfig(
                ticker=ticker,
                mae=mae,
                config=config_data,
                timestamp=timestamp,
                training_time=training_time
            )
            
            self.optimized_configs[ticker] = optimized_config
            logger.info(f"✅ Loaded optimized config for {ticker} (MAE: {mae:.4f})")
            return optimized_config
            
        except Exception as e:
            logger.error(f"❌ Error loading config for {ticker}: {e}")
            return None
    
    def get_best_config_for_ticker(self, ticker: str) -> Optional[Dict]:
        """Get the best configuration for a ticker"""
        if ticker in self.optimized_configs:
            return self.optimized_configs[ticker].config
        return None
    
    def apply_config_to_main(self, ticker: str) -> bool:
        """Apply optimized configuration to main.py"""
        config = self.get_best_config_for_ticker(ticker)
        
        if not config:
            logger.warning(f"No optimized config available for {ticker}")
            return False
        
        try:
            # Update the ARCHITECTURE_CONFIG in main.py
            self.update_main_config(config)
            logger.info(f"✅ Applied optimized config for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error applying config for {ticker}: {e}")
            return False
    
    def update_main_config(self, config: Dict):
        """Update the ARCHITECTURE_CONFIG in main.py"""
        main_file = 'main.py'
        
        if not os.path.exists(main_file):
            logger.error("main.py not found")
            return
        
        # Read main.py
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Find and replace ARCHITECTURE_CONFIG
        start_marker = "ARCHITECTURE_CONFIG = {"
        end_marker = "}"
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            logger.error("Could not find ARCHITECTURE_CONFIG in main.py")
            return
        
        # Find the end of the config block
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        # Create new config string
        new_config_str = f"""ARCHITECTURE_CONFIG = {{
    'type': 'hybrid',  # OPTIMIZED CONFIGURATION
    'cnn_filters': {config.get('cnn_filters', [64, 128])},
    'cnn_kernels': {config.get('cnn_kernels', [3, 5])},
    'lstm_units': {config.get('lstm_units', [50, 100, 50])},
    'transformer_heads': {config.get('transformer_heads', 8)},
    'transformer_key_dim': {config.get('transformer_key_dim', 64)},
    'dense_units': {config.get('dense_units', [128, 64])},
    'dropout_rate': {config.get('dropout_rate', 0.2)},
    'batch_size': {config.get('batch_size', 16)},
    'epochs': {config.get('epochs', 150)},
    'learning_rate': {config.get('learning_rate', 0.001)},
    'weight_decay': {config.get('weight_decay', 0.01)}
}}"""
        
        # Replace the config
        new_content = content[:start_idx] + new_config_str + content[end_idx:]
        
        # Write back to main.py
        with open(main_file, 'w') as f:
            f.write(new_content)
        
        logger.info("✅ Updated ARCHITECTURE_CONFIG in main.py")
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all tickers with optimized configs"""
        return list(self.optimized_configs.keys())
    
    def get_config_summary(self) -> str:
        """Get summary of all optimized configurations"""
        if not self.optimized_configs:
            return "No optimized configurations available."
        
        summary = "🏆 OPTIMIZED CONFIGURATIONS SUMMARY\n"
        summary += "="*50 + "\n"
        
        for ticker, config in self.optimized_configs.items():
            summary += f"\n{ticker}:\n"
            summary += f"  MAE: {config.mae:.4f}\n"
            summary += f"  Timestamp: {config.timestamp}\n"
            summary += f"  Training Time: {config.training_time:.1f}s\n"
        
        return summary

def auto_apply_best_config(ticker: str) -> bool:
    """Automatically apply the best configuration for a ticker"""
    config_manager = ConfigManager()
    return config_manager.apply_config_to_main(ticker)

def get_optimized_config(ticker: str) -> Optional[Dict]:
    """Get optimized configuration for a ticker"""
    config_manager = ConfigManager()
    return config_manager.get_best_config_for_ticker(ticker)

if __name__ == "__main__":
    print("🔧 CONFIGURATION MANAGER")
    print("="*30)
    
    config_manager = ConfigManager()
    
    # Show summary
    print(config_manager.get_config_summary())
    
    # List available tickers
    tickers = config_manager.get_all_tickers()
    print(f"\n📋 Available tickers: {', '.join(tickers) if tickers else 'None'}")
    
    # Example: Apply config for BTC-USD
    if 'BTC-USD' in tickers:
        print(f"\n🎯 Applying optimized config for BTC-USD...")
        success = config_manager.apply_config_to_main('BTC-USD')
        print(f"✅ Success: {success}")