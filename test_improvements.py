#!/usr/bin/env python3
"""
Test script to compare OLD vs NEW (IMPROVED) approaches.

This script runs both approaches on a single ticker and compares:
- Training time
- Model size (parameters)
- MAE, RMSE, MAPE
- Directional accuracy
- Baseline comparisons

Usage:
    python test_improvements.py
    python test_improvements.py BTC-USD  # Specify ticker
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Import from main.py
from main import (
    download_ticker_data_with_retry,
    add_technical_features,
    select_features_for_training,
    split_data_proper,
    create_dataset,
    split_data,
    create_hybrid_cnn_lstm_transformer_model,
    create_efficient_gru_model,
    create_simple_lstm_baseline,
    naive_forecast_baseline,
    moving_average_baseline,
    calculate_directional_accuracy,
    get_scaler,
    huber_loss,
    evalModel
)

from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import AdamW, Adam
from keras.callbacks import EarlyStopping


def test_old_approach(data, ticker):
    """
    Test OLD approach (current main.py logic).

    Features:
    - Only closing prices
    - MinMaxScaler fit on ALL data (data leakage!)
    - 80/20 split with validation_split=0.2
    - Hybrid CNN-LSTM-Transformer (1.1M params)
    - MSE loss

    Returns:
        dict with results
    """
    print("\n" + "="*80)
    print("🔴 TESTING OLD APPROACH (Current)")
    print("="*80)

    start_time = time.time()

    # OLD: Only closing prices
    closing_prices = data['Close'].values.reshape(-1, 1)
    print(f"📊 Features: 1 (Close only)")

    # OLD: Fit scaler on ALL data (DATA LEAKAGE!)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)
    print(f"⚠️  Scaler: MinMaxScaler fit on ALL data (leakage)")

    # Create dataset and split
    X, y = create_dataset(scaled_data)
    X_train, y_train, X_test, y_test = split_data(X, y, train_size=0.8)
    print(f"📊 Split: 80/20 train/test (no explicit val set)")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # OLD: Hybrid model (1.1M parameters)
    print(f"\n🏗️  Building Hybrid CNN-LSTM-Transformer...")
    model = create_hybrid_cnn_lstm_transformer_model(
        input_shape=(X_train.shape[1], 1),
        architecture_type='hybrid'
    )

    # Compile with MSE loss
    optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    param_count = model.count_params()
    print(f"📊 Parameters: {param_count:,}")

    # Train
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\n🚀 Training (validation_split=0.2)...")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced for testing
        batch_size=16,
        validation_split=0.2,  # OLD: carve validation from training
        callbacks=[early_stopping],
        verbose=0
    )

    training_time = time.time() - start_time

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Inverse transform for metrics
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mae, rmse, smape, mase, mape = evalModel(model, X_test, y_test, y_train)
    dir_acc = calculate_directional_accuracy(y_test_inv, y_pred_inv)

    # Baselines
    naive_mae = naive_forecast_baseline(y_train, y_test)
    ma_mae = moving_average_baseline(y_train, y_test, window=5)

    print(f"\n📊 Results:")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Parameters: {param_count:,}")
    print(f"   MAE: {mae*100:.2f}%")
    print(f"   MAPE: {mape*100:.2f}%")
    print(f"   Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"   Naive baseline MAE: {naive_mae:.4f}")
    print(f"   Improvement over naive: {((naive_mae - mae)/naive_mae)*100:.1f}%")

    return {
        'approach': 'OLD',
        'features': 1,
        'scaler': 'MinMaxScaler (leakage)',
        'model': 'Hybrid CNN-LSTM-Transformer',
        'parameters': param_count,
        'training_time': training_time,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'dir_acc': dir_acc,
        'naive_mae': naive_mae,
        'ma_mae': ma_mae,
        'improvement_vs_naive': ((naive_mae - mae)/naive_mae)*100
    }


def test_new_approach(data, ticker):
    """
    Test NEW (IMPROVED) approach.

    Features:
    - OHLCV + 14 technical indicators
    - StandardScaler fit ONLY on train data (no leakage!)
    - 70/15/15 proper train/val/test split
    - Efficient GRU model (300K params)
    - Huber loss (robust to outliers)

    Returns:
        dict with results
    """
    print("\n" + "="*80)
    print("🟢 TESTING NEW APPROACH (Improved)")
    print("="*80)

    start_time = time.time()

    # NEW: Add technical features
    data_with_features = add_technical_features(data)
    feature_cols = select_features_for_training(use_full_features=True)
    feature_data = data_with_features[feature_cols].values
    print(f"📊 Features: {len(feature_cols)} (OHLCV + indicators)")
    print(f"   {', '.join(feature_cols[:5])}...")

    # NEW: Split BEFORE scaling (no leakage!)
    train_data, val_data, test_data = split_data_proper(feature_data)
    print(f"✅ Scaler: StandardScaler fit ONLY on train data")

    # NEW: StandardScaler
    scaler = get_scaler('standard')
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Create datasets
    X_train, y_train = create_dataset(train_scaled)
    X_val, y_val = create_dataset(val_scaled)
    X_test, y_test = create_dataset(test_scaled)
    print(f"📊 Split: 70/15/15 train/val/test")
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # NEW: Efficient GRU model (300K parameters)
    print(f"\n🏗️  Building Efficient GRU Model...")
    model = create_efficient_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Compile with Huber loss
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=huber_loss(delta=0.5),  # NEW: Robust to outliers
        metrics=['mean_absolute_error']
    )

    param_count = model.count_params()
    print(f"📊 Parameters: {param_count:,}")

    # Train
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\n🚀 Training (explicit validation set)...")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced for testing
        batch_size=16,
        validation_data=(X_val, y_val),  # NEW: explicit validation
        callbacks=[early_stopping],
        verbose=0
    )

    training_time = time.time() - start_time

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Need to inverse transform with careful index handling
    # For multi-feature, y corresponds to Close (index 3)
    test_close_idx = 3  # Close is at index 3 in OHLCV

    # Create dummy array for inverse transform
    dummy_test = np.zeros((len(y_test), len(feature_cols)))
    dummy_test[:, test_close_idx] = y_test
    y_test_inv = scaler.inverse_transform(dummy_test)[:, test_close_idx]

    dummy_pred = np.zeros((len(y_pred), len(feature_cols)))
    dummy_pred[:, test_close_idx] = y_pred
    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, test_close_idx]

    # Calculate metrics in original scale
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

    mae_abs = mean_absolute_error(y_test_inv, y_pred_inv)
    mae_pct = mae_abs / np.mean(y_test_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    dir_acc = calculate_directional_accuracy(y_test_inv, y_pred_inv)

    # Baselines (in scaled space)
    naive_mae = naive_forecast_baseline(y_train, y_test)
    ma_mae = moving_average_baseline(y_train, y_test, window=5)

    print(f"\n📊 Results:")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Parameters: {param_count:,}")
    print(f"   MAE: {mae_pct*100:.2f}%")
    print(f"   MAPE: {mape*100:.2f}%")
    print(f"   Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"   Naive baseline MAE: {naive_mae:.4f}")
    print(f"   Improvement over naive: {((naive_mae - mae_pct)/naive_mae)*100:.1f}%")

    return {
        'approach': 'NEW',
        'features': len(feature_cols),
        'scaler': 'StandardScaler (no leakage)',
        'model': 'Efficient GRU',
        'parameters': param_count,
        'training_time': training_time,
        'mae': mae_pct,
        'rmse': rmse / np.mean(y_test_inv),  # As percentage
        'mape': mape,
        'dir_acc': dir_acc,
        'naive_mae': naive_mae,
        'ma_mae': ma_mae,
        'improvement_vs_naive': ((naive_mae - mae_pct)/naive_mae)*100
    }


def print_comparison(old_results, new_results):
    """Print side-by-side comparison of results."""
    print("\n" + "="*80)
    print("📊 COMPARISON: OLD vs NEW")
    print("="*80)

    print(f"\n{'Metric':<30} {'OLD':<25} {'NEW':<25} {'Change':<15}")
    print("-" * 95)

    # Features
    print(f"{'Features':<30} {old_results['features']:<25} {new_results['features']:<25} "
          f"{'+' + str(new_results['features'] - old_results['features']):<15}")

    # Model
    print(f"{'Model':<30} {old_results['model'][:23]:<25} {new_results['model'][:23]:<25} {'-':<15}")

    # Parameters
    old_params = old_results['parameters']
    new_params = new_results['parameters']
    param_reduction = ((old_params - new_params) / old_params) * 100
    print(f"{'Parameters':<30} {f'{old_params:,}':<25} {f'{new_params:,}':<25} "
          f"{f'-{param_reduction:.0f}%':<15}")

    # Training time
    old_time = old_results['training_time']
    new_time = new_results['training_time']
    time_improvement = ((old_time - new_time) / old_time) * 100
    print(f"{'Training Time (s)':<30} {f'{old_time:.1f}':<25} {f'{new_time:.1f}':<25} "
          f"{f'-{time_improvement:.0f}%':<15}")

    # MAE
    old_mae = old_results['mae'] * 100
    new_mae = new_results['mae'] * 100
    mae_improvement = ((old_mae - new_mae) / old_mae) * 100
    print(f"{'MAE (%)':<30} {f'{old_mae:.2f}':<25} {f'{new_mae:.2f}':<25} "
          f"{f'-{mae_improvement:.0f}%' if mae_improvement > 0 else f'+{abs(mae_improvement):.0f}%':<15}")

    # MAPE
    old_mape = old_results['mape'] * 100
    new_mape = new_results['mape'] * 100
    mape_improvement = ((old_mape - new_mape) / old_mape) * 100
    print(f"{'MAPE (%)':<30} {f'{old_mape:.2f}':<25} {f'{new_mape:.2f}':<25} "
          f"{f'-{mape_improvement:.0f}%' if mape_improvement > 0 else f'+{abs(mape_improvement):.0f}%':<15}")

    # Directional Accuracy
    old_dir = old_results['dir_acc'] * 100
    new_dir = new_results['dir_acc'] * 100
    dir_improvement = new_dir - old_dir
    print(f"{'Directional Acc (%)':<30} {f'{old_dir:.1f}':<25} {f'{new_dir:.1f}':<25} "
          f"{f'+{dir_improvement:.1f}%':<15}")

    # Improvement vs naive
    old_imp = old_results['improvement_vs_naive']
    new_imp = new_results['improvement_vs_naive']
    print(f"{'Beat Naive By (%)':<30} {f'{old_imp:.1f}':<25} {f'{new_imp:.1f}':<25} "
          f"{f'+{new_imp - old_imp:.1f}%':<15}")

    print("\n" + "="*80)
    print("✅ SUMMARY")
    print("="*80)

    if new_mae < old_mae:
        print(f"✅ MAE improved by {mae_improvement:.1f}% (lower is better)")
    else:
        print(f"⚠️  MAE increased by {abs(mae_improvement):.1f}% (may be due to no data leakage)")

    if new_time < old_time:
        print(f"✅ Training {time_improvement:.0f}% faster")
    else:
        print(f"⚠️  Training {abs(time_improvement):.0f}% slower")

    if new_params < old_params:
        print(f"✅ {param_reduction:.0f}% fewer parameters")

    if new_dir > old_dir:
        print(f"✅ Directional accuracy improved by {dir_improvement:.1f}%")

    print(f"✅ No data leakage (valid metrics)")
    print(f"✅ Using {new_results['features']}x more features")


def main():
    """Main test function."""
    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else '^IXIC'

    print("\n" + "="*80)
    print(f"🧪 TESTING IMPROVEMENTS ON: {ticker}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Download data
    print(f"\n📥 Downloading data for {ticker}...")
    data = download_ticker_data_with_retry(ticker, period='max')

    if data is None or len(data) < 100:
        print(f"❌ Failed to download sufficient data for {ticker}")
        return

    print(f"✅ Downloaded {len(data)} days of data")

    # Test both approaches
    old_results = test_old_approach(data, ticker)
    new_results = test_new_approach(data, ticker)

    # Compare
    print_comparison(old_results, new_results)

    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
