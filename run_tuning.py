#!/usr/bin/env python3
"""
Simple script to run fine-tuning for specific tickers
Usage: python run_tuning.py --ticker BTC-USD --configs 30
"""

import argparse
import sys
import os
from datetime import datetime
from fine_tuning_script import HyperparameterTuner, schedule_tuning_job

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for stock forecasting')
    parser.add_argument('--ticker', type=str, default='BTC-USD', 
                       help='Ticker symbol to tune (default: BTC-USD)')
    parser.add_argument('--configs', type=int, default=20, 
                       help='Number of configurations to test (default: 20)')
    parser.add_argument('--workers', type=int, default=2, 
                       help='Number of parallel workers (default: 2)')
    parser.add_argument('--period', type=str, default='max', 
                       help='Data period for training (default: max - all available data)')
    parser.add_argument('--apply', action='store_true', 
                       help='Apply best config to main.py after tuning')
    
    args = parser.parse_args()
    
    print("🧪 HYPERPARAMETER TUNING RUNNER")
    print("="*40)
    print(f"Ticker: {args.ticker}")
    print(f"Configurations: {args.configs}")
    print(f"Workers: {args.workers}")
    print(f"Period: {args.period}")
    print(f"Apply Best Config: {args.apply}")
    print("="*40)
    
    try:
        # Create tuner
        tuner = HyperparameterTuner(ticker=args.ticker, test_period=args.period)
        
        # Run tuning
        print(f"\n🚀 Starting tuning for {args.ticker}...")
        results = tuner.run_hyperparameter_search(
            max_configs=args.configs, 
            max_workers=args.workers
        )
        
        # Show results
        if results:
            best_result = results[0]
            print(f"\n🏆 BEST RESULT:")
            print(f"MAE: {best_result.mae:.4f}")
            print(f"RMSE: {best_result.rmse:.4f}")
            print(f"MAPE: {best_result.mape:.4f}")
            print(f"Training Time: {best_result.training_time:.1f}s")
            
            # Apply best config if requested
            if args.apply:
                print(f"\n🎯 Applying best configuration...")
                tuner.apply_best_config_to_main()
                print("✅ Best configuration applied to main.py")
            
            # Generate report
            report = tuner.generate_tuning_report()
            print(f"\n{report}")
            
        else:
            print("❌ No results obtained")
            return 1
            
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        return 1
    
    print("\n✅ Tuning completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())