import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, AdditiveAttention, Permute, Reshape, Multiply, Attention, Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from telegram_sender import send_telegram, send_image_to_telegram
import asyncio
import os
import pickle
from keras.models import load_model
import requests
from textblob import TextBlob
import tweepy
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Market Sentiment Analysis Functions
def get_fear_greed_index():
    """
    Fetch Fear & Greed Index from CNN (alternative free API)
    Returns a value between 0-100 where 0 = Extreme Fear, 100 = Extreme Greed
    """
    try:
        # Alternative Fear & Greed Index API
        url = "https://api.alternative.me/fng/?limit=10"
        response = requests.get(url)
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            current_fng = float(data['data'][0]['value'])
            return current_fng / 100.0  # Normalize to 0-1 range
        else:
            return 0.5  # Neutral if no data
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return 0.5  # Return neutral sentiment

def get_vix_data():
    """
    Fetch VIX (Volatility Index) data as a market sentiment indicator
    High VIX = High fear/volatility, Low VIX = Low fear/complacency
    """
    try:
        vix = yf.download('^VIX', period='1mo', interval='1d')
        if not vix.empty:
            latest_vix = vix['Close'].iloc[-1]
            # Normalize VIX (typically ranges 10-80, extreme values can go higher)
            normalized_vix = min(max((latest_vix - 10) / 70, 0), 1)
            return 1 - normalized_vix  # Invert so high VIX = low sentiment score
        else:
            return 0.5
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return 0.5

def get_put_call_ratio():
    """
    Simulate Put/Call ratio analysis
    In a real implementation, you'd fetch this from options data
    """
    try:
        # For demonstration, we'll use VIX as a proxy
        # In practice, you'd fetch actual put/call ratio data
        vix = yf.download('^VIX', period='5d', interval='1d')
        if not vix.empty and len(vix) >= 2:
            vix_change = (vix['Close'].iloc[-1] - vix['Close'].iloc[-2]) / vix['Close'].iloc[-2]
            # Convert change to sentiment (negative VIX change = positive sentiment)
            sentiment = 0.5 + (-vix_change * 0.5)
            return max(min(sentiment, 1), 0)
        return 0.5
    except Exception as e:
        print(f"Error calculating put/call ratio proxy: {e}")
        return 0.5

def get_market_breadth_sentiment():
    """
    Analyze market breadth using major indices
    Compare performance of different market segments
    """
    try:
        indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, NASDAQ
        sentiment_scores = []
        
        for index in indices:
            data = yf.download(index, period='5d', interval='1d')
            if not data.empty and len(data) >= 2:
                # Calculate 5-day performance
                performance = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                # Convert to sentiment score (0-1)
                sentiment = 0.5 + (performance * 2)  # Amplify the signal
                sentiment_scores.append(max(min(sentiment, 1), 0))
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.5
    except Exception as e:
        print(f"Error calculating market breadth: {e}")
        return 0.5

def get_crypto_sentiment():
    """
    Get cryptocurrency market sentiment using Bitcoin dominance and performance
    """
    try:
        btc = yf.download('BTC-USD', period='7d', interval='1d')
        if not btc.empty and len(btc) >= 2:
            # Calculate 7-day performance
            btc_performance = (btc['Close'].iloc[-1] - btc['Close'].iloc[0]) / btc['Close'].iloc[0]
            # Convert to sentiment (crypto often leads market sentiment)
            sentiment = 0.5 + (btc_performance * 1.5)
            return max(min(sentiment, 1), 0)
        return 0.5
    except Exception as e:
        print(f"Error fetching crypto sentiment: {e}")
        return 0.5

def calculate_composite_sentiment():
    """
    Calculate a composite sentiment score from multiple indicators
    """
    print("Calculating market sentiment indicators...")
    
    # Get individual sentiment scores
    fear_greed = get_fear_greed_index()
    vix_sentiment = get_vix_data()
    put_call_sentiment = get_put_call_ratio()
    breadth_sentiment = get_market_breadth_sentiment()
    crypto_sentiment = get_crypto_sentiment()
    
    # Weights for different sentiment indicators
    weights = {
        'fear_greed': 0.3,
        'vix': 0.25,
        'put_call': 0.2,
        'breadth': 0.15,
        'crypto': 0.1
    }
    
    # Calculate weighted composite sentiment
    composite = (
        fear_greed * weights['fear_greed'] +
        vix_sentiment * weights['vix'] +
        put_call_sentiment * weights['put_call'] +
        breadth_sentiment * weights['breadth'] +
        crypto_sentiment * weights['crypto']
    )
    
    print(f"Sentiment Indicators:")
    print(f"  Fear & Greed Index: {fear_greed:.3f}")
    print(f"  VIX Sentiment: {vix_sentiment:.3f}")
    print(f"  Put/Call Sentiment: {put_call_sentiment:.3f}")
    print(f"  Market Breadth: {breadth_sentiment:.3f}")
    print(f"  Crypto Sentiment: {crypto_sentiment:.3f}")
    print(f"  Composite Sentiment: {composite:.3f}")
    
    return composite, {
        'fear_greed': fear_greed,
        'vix': vix_sentiment,
        'put_call': put_call_sentiment,
        'breadth': breadth_sentiment,
        'crypto': crypto_sentiment,
        'composite': composite
    }

# Enhanced dataset creation with sentiment
def create_dataset_with_sentiment(data, sentiment_score, days_range=60):
    """
    Create dataset with sentiment features
    """
    X, y = [], []
    for i in range(days_range, len(data)):
        # Original price features
        price_features = data[i - days_range:i, 0]
        
        # Add sentiment as additional feature (repeated for the sequence)
        sentiment_features = np.full(days_range, sentiment_score)
        
        # Combine price and sentiment features
        combined_features = np.column_stack([price_features, sentiment_features])
        
        X.append(combined_features)
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Enhanced model building with sentiment
def build_sentiment_enhanced_model(input_shape):
    """
    Build LSTM model that can handle both price and sentiment features
    """
    model = Sequential()

    # Bidirectional LSTM layers with enhanced input shape
    model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=input_shape)))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=150, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))

    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(LSTM(units=50, return_sequences=True))

    model.add(Flatten())

    # Additional dense layers to process sentiment information
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=50, activation='relu'))

    # Final Dense Layer
    model.add(Dense(units=1))

    return model

# Original functions (keeping them for compatibility)
def create_dataset(data, days_range=60):
    X, y = [], []
    for i in range(days_range, len(data)):
        X.append(data[i - days_range:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def split_data(X, y, train_size=0.8):
    split = int(train_size * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Handle both 2D and 3D input shapes
    if len(X_train.shape) == 2:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, X_test, y_test

def split_data_with_sentiment(X, y, train_size=0.8):
    """
    Split data for sentiment-enhanced model
    """
    split = int(train_size * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train, y_train = np.array(X_train), np.array(y_train)
    
    return X_train, y_train, X_test, y_test

def smape(X, y):
    X = np.array(X)
    y = np.array(y)
    return np.mean(np.abs((X - y) / ((np.abs(X) + np.abs(y)) / 2)))

def mase(y_true, y_pred, y_train):
    mae_train = np.mean(np.abs(y_train - np.mean(y_train)))
    mae_test = np.mean(np.abs(y_true - y_pred))
    return mae_test / mae_train

def save_model_and_scaler(model, scaler, ticker, mae_value, sentiment_data=None):
    """Save model, scaler, and sentiment data with MAE information"""
    os.makedirs('models', exist_ok=True)
    
    clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
    
    # Save model
    model_path = f'models/{clean_ticker}_sentiment_model.h5'
    model.save(model_path)
    
    # Save scaler
    scaler_path = f'models/{clean_ticker}_sentiment_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save MAE info and sentiment data
    mae_info_path = f'models/{clean_ticker}_sentiment_mae_info.pkl'
    mae_info = {
        'mae': mae_value, 
        'ticker': ticker,
        'sentiment_data': sentiment_data,
        'model_type': 'sentiment_enhanced'
    }
    with open(mae_info_path, 'wb') as f:
        pickle.dump(mae_info, f)
    
    print(f"Sentiment-enhanced model saved for {ticker} with MAE: {mae_value:.4f}")

def load_model_and_scaler(ticker, sentiment_enhanced=True):
    """Load model and scaler if they exist and MAE is <= 10%"""
    clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
    
    if sentiment_enhanced:
        model_path = f'models/{clean_ticker}_sentiment_model.h5'
        scaler_path = f'models/{clean_ticker}_sentiment_scaler.pkl'
        mae_info_path = f'models/{clean_ticker}_sentiment_mae_info.pkl'
    else:
        model_path = f'models/{clean_ticker}_model.h5'
        scaler_path = f'models/{clean_ticker}_scaler.pkl'
        mae_info_path = f'models/{clean_ticker}_mae_info.pkl'
    
    if not all(os.path.exists(path) for path in [model_path, scaler_path, mae_info_path]):
        return None, None, None
    
    try:
        with open(mae_info_path, 'rb') as f:
            mae_info = pickle.load(f)
        
        if mae_info['mae'] > 0.10:
            print(f"Saved model for {ticker} has MAE {mae_info['mae']:.4f} > 10%, will retrain")
            return None, None, None
        
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Loaded existing sentiment-enhanced model for {ticker} with MAE: {mae_info['mae']:.4f}")
        return model, scaler, mae_info['mae']
        
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None, None, None

def evalModel(model, X_test, y_test, y_train):
    """Evaluate model and return metrics"""
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Handle both 2D and 3D input shapes
    if len(X_test.shape) == 2:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss: ", test_loss)

    y_pred = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    smape_value = smape(y_test, y_pred)
    mase_value = mase(y_test, y_pred, y_train)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Mean Absolute Error: ", mae)
    print("Root Mean Square Error: ", rmse)
    print("Symmetric Mean Absolute Percentage Error: ", smape_value)
    print("Mean Absolute Scaled Error: ", mase_value)
    print("Mean Absolute Percentage Error: ", mape)
    
    return mae, rmse, smape_value, mase_value, mape

# Main execution
if __name__ == "__main__":
    
    # Calculate market sentiment once for all tickers
    composite_sentiment, sentiment_breakdown = calculate_composite_sentiment()
    
    for Ticker in ['MXN=X', '^MXX', 'BTC-USD', 'ETH-USD', 'PAXG-USD', '^IXIC', '^SP500-45']:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {Ticker} with Sentiment Analysis...")
            print(f"Current Market Sentiment: {composite_sentiment:.3f}")
            print(f"{'='*50}")
            
            # Download the data (using maximum available historical data)
            data = yf.download(Ticker, period='max', interval='1d', timeout=20)
            
            # Check if data is empty
            if data.empty or len(data) < 60:
                print(f"Skipping {Ticker}: Insufficient data (got {len(data)} days, need at least 60)")
                continue
                
            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            mae = None
            rmse = None
            smape_value = None
            mase_value = None
            mape = None

            # Create dataset with sentiment features
            X, y = create_dataset_with_sentiment(scaled_data, composite_sentiment)
            X_train, y_train, X_test, y_test = split_data_with_sentiment(X, y)

            # Try to load existing sentiment-enhanced model first
            model, loaded_scaler, loaded_mae = load_model_and_scaler(Ticker, sentiment_enhanced=True)
            
            if model is not None:
                # Use loaded model and scaler
                scaler = loaded_scaler
                mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
            else:
                # Build and train new sentiment-enhanced model
                print("Training new sentiment-enhanced model...")
                model = build_sentiment_enhanced_model((X_train.shape[1], X_train.shape[2]))

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error',
                              metrics=['mean_absolute_error', 'mean_squared_error'])

                # Train the model
                history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                          validation_split=0.2, callbacks=[early_stopping], verbose=1)

                print(model.summary())

                mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
                
                # Save model if MAE is <= 10%
                if mae <= 0.10:
                    save_model_and_scaler(model, scaler, Ticker, mae, sentiment_breakdown)
                else:
                    print(f"Model MAE {mae:.4f} > 10%, not saving model")

            # Make predictions with sentiment
            data_for_prediction = data.iloc[-60:]
            closing_prices = data_for_prediction['Close'].values
            
            # Scale the prediction data
            prediction_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prediction_data = prediction_scaler.fit_transform(closing_prices.reshape(-1, 1))

            # Predict the next 10 days iteratively
            predicted_prices = []
            current_batch = scaled_prediction_data[-60:].reshape(1, 60, 1)
            
            # Add sentiment to the current batch
            sentiment_batch = np.full((1, 60, 1), composite_sentiment)
            current_batch_with_sentiment = np.concatenate([current_batch, sentiment_batch], axis=2)

            for i in range(10):
                # Get the prediction (next day)
                next_prediction = model.predict(current_batch_with_sentiment, verbose=0)

                # Update the batch for next prediction
                next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
                sentiment_next = np.full((1, 1, 1), composite_sentiment)
                next_with_sentiment = np.concatenate([next_prediction_reshaped, sentiment_next], axis=2)
                
                # Slide the window
                current_batch_with_sentiment = np.concatenate([
                    current_batch_with_sentiment[:, 1:, :], 
                    next_with_sentiment
                ], axis=1)

                # Inverse transform the prediction to the original price scale
                predicted_prices.append(prediction_scaler.inverse_transform(next_prediction)[0, 0])

            print("Predicted Stock Prices for the next 10 days: ", predicted_prices)

            # Create prediction visualization
            last_date = data_for_prediction.index[-1]
            next_day = last_date + pd.Timedelta(days=1)
            prediction_dates = pd.date_range(start=next_day, periods=10)

            predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

            # Create enhanced plot with sentiment information
            plt.figure(figsize=(12, 8))
            
            # Plot historical data
            plt.subplot(2, 1, 1)
            plt.plot(data_for_prediction.index[-60:], data_for_prediction['Close'][-60:], 
                    linestyle='-', marker='o', color='blue', label='Historical Data', markersize=3)
            plt.plot(prediction_dates, predicted_prices, 
                    linestyle='-', marker='o', color='red', label='Predictions', markersize=4)
            
            # Update ticker names for display
            display_ticker = Ticker
            if Ticker == '^IXIC':
                display_ticker = 'NASDAQ Composite'
            elif Ticker == '^MXX':
                display_ticker = 'IPC MEXICO'
            elif Ticker == 'MXN=X':
                display_ticker = 'USD/MXN'
            elif Ticker == '^SP500-45':
                display_ticker = 'S&P 500 - Information Technology'

            plt.title(f"{display_ticker} - Price Prediction with Market Sentiment")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Plot sentiment indicators
            plt.subplot(2, 1, 2)
            sentiment_labels = ['Fear&Greed', 'VIX', 'Put/Call', 'Breadth', 'Crypto', 'Composite']
            sentiment_values = [
                sentiment_breakdown['fear_greed'],
                sentiment_breakdown['vix'],
                sentiment_breakdown['put_call'],
                sentiment_breakdown['breadth'],
                sentiment_breakdown['crypto'],
                sentiment_breakdown['composite']
            ]
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'black']
            bars = plt.bar(sentiment_labels, sentiment_values, color=colors, alpha=0.7)
            plt.title('Market Sentiment Indicators')
            plt.ylabel('Sentiment Score (0-1)')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, sentiment_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the enhanced plot
            clean_ticker = Ticker.replace('/', '_').replace('^', '').replace('=', '_')
            image_name_full = f'sentiment_enhanced_{clean_ticker}_predictions.png'
            plt.savefig(image_name_full, dpi=300, bbox_inches='tight')
            plt.close()

            # Send enhanced telegram message with sentiment data
            sentiment_text = "\n".join([f"  • {k.title().replace('_', ' ')}: {v:.3f}" 
                                      for k, v in sentiment_breakdown.items()])
            
            message = f'''Here are the next 10 days predictions for <b>{display_ticker}</b> with Market Sentiment Analysis.

<b>📈 Predicted Prices:</b> {[f"{p:.2f}" for p in predicted_prices]}

<b>🎯 Model Performance:</b>
  • Mean Absolute Error: <b>{mae*100:.2f}%</b>
  • Mean Absolute Percentage Error: <b>{mape*100:.2f}%</b>
  • Mean Absolute Scaled Error: <b>{mase_value:.2f}</b>
  • Symmetric Mean Absolute Percentage Error: <b>{smape_value*100:.2f}%</b>
  • Root Mean Square Error: <b>{rmse*100:.2f}%</b>

<b>📊 Market Sentiment Indicators:</b>
{sentiment_text}

<b>🔮 Sentiment Impact:</b> The current composite sentiment score of <b>{composite_sentiment:.3f}</b> suggests {"bullish" if composite_sentiment > 0.6 else "bearish" if composite_sentiment < 0.4 else "neutral"} market conditions.'''

            asyncio.run(send_telegram(message))
            asyncio.run(send_image_to_telegram(image_name_full, 
                       caption=f'Sentiment-Enhanced Predictions for {display_ticker}'))

        except Exception as e:
            print(f"Error processing {Ticker}: {e}")
            continue

    print("\n" + "="*50)
    print("SENTIMENT-ENHANCED LSTM FORECASTING COMPLETE")
    print("="*50)
