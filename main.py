import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Bidirectional,  Dropout, AdditiveAttention, Permute, Reshape, Multiply, Attention, Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from telegram_sender import send_telegram, send_image_to_telegram
from sentiment_analyzer import get_ticker_sentiment, format_sentiment_for_telegram
import asyncio
import os
import pickle
import time
import random
from datetime import datetime
from keras.models import load_model


def download_ticker_data_with_retry(ticker, period='6y', max_retries=5, base_delay=10):
    """
    Download ticker data with exponential backoff retry logic
    
    Args:
        ticker (str): Ticker symbol
        period (str): Period for data download
        max_retries (int): Maximum number of retry attempts
        base_delay (int): Base delay in seconds for exponential backoff
    
    Returns:
        pandas.DataFrame: Downloaded data or None if failed
    """
    print(f"📥 Downloading data for {ticker}...")
    
    for attempt in range(max_retries):
        try:
            # Add random jitter to avoid synchronized requests
            if attempt > 0:
                jitter = random.uniform(0.5, 1.5)
                delay = (base_delay * (2 ** (attempt - 1))) * jitter
                print(f"⏳ Attempt {attempt + 1}/{max_retries} - Waiting {delay:.1f} seconds...")
                time.sleep(delay)
            
            # Download data with timeout
            data = yf.download(
                ticker, 
                period=period, 
                interval='1d', 
                timeout=30,
                progress=False,  # Disable progress bar to reduce output
                auto_adjust=True,  # Explicitly set to avoid FutureWarning
                prepost=False,  # Don't include pre/post market data
                threads=False   # Disable threading for better error handling
            )
            
            # Check if data is valid
            if data.empty:
                print(f"⚠️  Empty data received for {ticker}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            
            if len(data) < 61:  # Need at least 61 days for 60-day lookback
                print(f"⚠️  Insufficient data for {ticker}: {len(data)} days (need 61+)")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            
            print(f"✅ Downloaded {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            error_message = str(e).lower()
            
            if "rate limit" in error_message or "too many requests" in error_message:
                print(f"🚫 Rate limited for {ticker} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(5, 15)
                    print(f"⏳ Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached for {ticker} due to rate limiting")
                    return None
            else:
                print(f"❌ Error downloading {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Short delay for other errors
                    continue
                else:
                    return None
    
    return None


def process_single_ticker(ticker, ticker_index, total_tickers):
    """
    Process a single ticker with proper error handling and delays
    
    Args:
        ticker (str): Ticker symbol to process
        ticker_index (int): Index of current ticker (for progress tracking)
        total_tickers (int): Total number of tickers
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"🔄 Processing {ticker} ({ticker_index + 1}/{total_tickers})")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # Download data with retry logic
        data = download_ticker_data_with_retry(ticker, period='6y', max_retries=5, base_delay=10)
        
        if data is None:
            print(f"❌ Failed to download data for {ticker} after all retries")
            return False
        
        # Add a small delay between data download and processing
        time.sleep(random.uniform(2, 5))
        
        # Continue with the rest of the processing...
        return process_ticker_data(ticker, data)
        
    except Exception as e:
        print(f"❌ Unexpected error processing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_ticker_data(ticker, data):
    """
    Process the downloaded ticker data (main processing logic)
    
    Args:
        ticker (str): Ticker symbol
        data (pandas.DataFrame): Downloaded market data
    
    Returns:
        bool: True if processing was successful
    """
    try:
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        mae = None
        rmse = None
        smape_value = None
        mase_value = None
        mape = None

        X, y = create_dataset(scaled_data)
        X_train, y_train, X_test, y_test = split_data(X, y)

        # Try to load existing model first
        model, loaded_scaler, loaded_mae = load_model_and_scaler(ticker)
        
        if model is not None:
            # Use loaded model and scaler
            scaler = loaded_scaler
            mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
        else:
            # Build and train new model
            model = Sequential()

            model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))))
            model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1))))
            model.add(Bidirectional(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1))))
            model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1))))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))))

            model.add(Dropout(0.2))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(LSTM(units=50, return_sequences=True))
            # The attention mechanism
            attention = AdditiveAttention(name="attention_weight")

            model.add(Flatten())

            # Final Dense Layer
            model.add(Dense(units=1))

            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            # compile the model
            model.compile(optimizer='adam', loss='mean_squared_error',
                          metrics=['mean_absolute_error', 'mean_squared_error'])

            # train the model
            model.fit(X_train, y_train, epochs=100, batch_size=32,
                      validation_split=0.2, callbacks=[early_stopping])

            print(model.summary())

            mae, rmse, smape_value, mase_value, mape = evalModel(model, X_test, y_test, y_train)
            
            # Save model if MAE is <= 10%
            if mae <= 0.10:  # 10% threshold
                save_model_and_scaler(model, scaler, ticker, mae)
            else:
                print(f"Model MAE {mae:.4f} > 10%, not saving model")

        # Analyze market sentiment
        print("📊 Analyzing market sentiment...")
        try:
            sentiment_data = get_ticker_sentiment(ticker)
            sentiment_message = format_sentiment_for_telegram(ticker, sentiment_data)
            print(f"Sentiment Analysis for {ticker}:")
            print(f"  Score: {sentiment_data['sentiment_score']:.3f}")
            print(f"  Label: {sentiment_data['sentiment_label']}")
            print(f"  Confidence: {sentiment_data['confidence']*100:.0f}%")
        except Exception as e:
            print(f"Error analyzing sentiment for {ticker}: {e}")
            sentiment_message = f"\n📊 <b>Sentiment Analysis for {ticker}</b>\n❌ Could not analyze sentiment\n"

        # Fetching the latest 60 days of stock data
        data = data.iloc[-60:]  # Get the last 60 days of data

        # Selecting the 'Close' price and converting to numpy array
        closing_prices = data['Close'].values

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

        # Predict the next 10 days iteratively
        predicted_prices = []
        current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

        for i in range(10):  # Predicting 10 days
            # Get the prediction (next day)
            next_prediction = model.predict(current_batch, verbose=0)

            # Handle different model output shapes
            try:
                # Try to reshape to (1, 1, 1) - this works for simple models
                next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
                prediction_value = next_prediction.flatten()[0]
            except ValueError:
                # If reshape fails, handle complex model outputs
                if len(next_prediction.shape) == 3 and next_prediction.shape[1] > 1:
                    # Take the last prediction from sequence models
                    prediction_value = next_prediction[0, -1, 0]
                    next_prediction_reshaped = np.array([[[prediction_value]]])
                else:
                    # For other shapes, take the first element
                    prediction_value = next_prediction.flatten()[0]
                    next_prediction_reshaped = np.array([[[prediction_value]]])

            # Append the prediction to the batch used for predicting
            current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

            # Inverse transform the prediction to the original price scale
            predicted_prices.append(scaler.inverse_transform([[prediction_value]])[0, 0])

        print("Predicted Stock Prices for the next 10 days: ", predicted_prices)

        # Creating a list of dates for the predictions
        last_date = data.index[-1]
        next_day = last_date + pd.Timedelta(days=1)
        prediction_dates = pd.date_range(start=next_day, periods=10)

        # Assuming 'predicted_prices' is your list of predicted prices for the next 4 days
        predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

        # # Overlaying the predicted data
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df.index, predictions_df['Close'], linestyle='dashed', marker='o', color='red')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xticks(rotation=90)

        # Save plot as image to send it to the user
        # Clean ticker name for file naming
        clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
        image_name = f'{clean_ticker}_predictions.png'
        plt.savefig(image_name)
        # plt.show()
        plt.close()

        # Send the image to Telegram# Creating a list of dates for the predictions
        last_date = data.index[-1]
        next_day = last_date + pd.Timedelta(days=1)
        prediction_dates = pd.date_range(start=next_day, periods=10)

        # Adding predictions to the DataFrame
        predicted_data = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

        # Combining both actual and predicted data
        combined_data = pd.concat([data['Close'], predicted_data['Close']])
        combined_data = combined_data[-64:] # Last 60 days of actual data + 4 days of predictions

        # Plotting the actual data
        plt.figure(figsize=(10,6))
        plt.plot(data.index[-60:], data['Close'][-60:], linestyle='-', marker='o', color='blue', label='Actual Data')

        # Plotting the predicted data
        plt.plot(prediction_dates, predicted_prices, linestyle='-', marker='o', color='red', label='Predicted Data')
        display_ticker = ticker
        if ticker == '^IXIC':
            display_ticker = 'NASDAQ Composite'
        elif ticker == '^MXX':
            display_ticker = 'IPC MEXICO'
        elif ticker == 'MXN=X':
            display_ticker = 'USD/MXN'
        elif ticker == '^SP500-45':
            display_ticker = 'S&P 500 - Information Technology'

        plt.title(f"{display_ticker} Stock Price: Last 60 Days and Next 4 Days Predicted")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        image_name_full = f'full {clean_ticker}_predictions.png'
        plt.savefig(image_name_full)
        plt.close()
        # plt.show()

        try:
            # Create comprehensive message with predictions and sentiment
            prediction_message = f'🔮 <b>LSTM Forecast Report for {display_ticker}</b>\n\n'
            prediction_message += f'📈 <b>Next 10 Days Predictions:</b>\n<code>{predicted_prices}</code>\n\n'
            prediction_message += f'📊 <b>Model Performance Metrics:</b>\n'
            prediction_message += f'• Mean Absolute Error: <b>{mae*100:.2f}%</b>\n'
            prediction_message += f'• Mean Absolute Percentage Error: <b>{mape*100:.2f}%</b>\n'
            prediction_message += f'• Mean Absolute Scaled Error: <b>{mase_value:.2f}</b>\n'
            prediction_message += f'• Symmetric MAPE: <b>{smape_value*100:.2f}%</b>\n'
            prediction_message += f'• Root Mean Square Error: <b>{rmse*100:.2f}%</b>\n'
            prediction_message += sentiment_message
            prediction_message += f'\n📸 Charts attached below ⬇️'
            
            asyncio.run(send_telegram(prediction_message))
            asyncio.run(send_image_to_telegram(image_name, caption=f'📊 {display_ticker} - Next 10 Days Prediction'))
            asyncio.run(send_image_to_telegram(image_name_full, caption=f'📈 {display_ticker} - Historical Data + Predictions'))
        except Exception as e:
            print(f"Error sending Telegram message for {ticker}: {e}")
            
        print(f"✅ Successfully processed {ticker}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing ticker data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False


# Create the dataset
def create_dataset(data, days_range=60):
    X, y = [], []
    for i in range(days_range, len(data)):
        X.append(data[i - days_range:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Split the data into training and testing sets
def split_data(X, y, train_size=0.8):
    split = int(train_size * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, X_test, y_test

def smape(X, y):
    X = np.array(X)
    y = np.array(y)
    return np.mean(np.abs((X - y) / ((np.abs(X) + np.abs(y)) / 2)))

def mase(y_true, y_pred, y_train):
    # Calculate the mean absolute error of the training data
    mae_train = np.mean(np.abs(y_train - np.mean(y_train)))
    # Calculate the mean absolute error of the test data
    mae_test = np.mean(np.abs(y_true - y_pred))
    # Calculate the MASE
    return mae_test / mae_train

def save_model_and_scaler(model, scaler, ticker, mae_value):
    """Save model and scaler with MAE information"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Clean ticker name for file naming
    clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
    
    # Save model
    model_path = f'models/{clean_ticker}_model.h5'
    model.save(model_path)
    
    # Save scaler
    scaler_path = f'models/{clean_ticker}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save MAE info
    mae_info_path = f'models/{clean_ticker}_mae_info.pkl'
    mae_info = {'mae': mae_value, 'ticker': ticker}
    with open(mae_info_path, 'wb') as f:
        pickle.dump(mae_info, f)
    
    print(f"Model saved for {ticker} with MAE: {mae_value:.4f}")

def load_model_and_scaler(ticker):
    """Load model and scaler if they exist and MAE is <= 10%"""
    # Clean ticker name for file naming
    clean_ticker = ticker.replace('/', '_').replace('^', '').replace('=', '_')
    
    model_path = f'models/{clean_ticker}_model.h5'
    scaler_path = f'models/{clean_ticker}_scaler.pkl'
    mae_info_path = f'models/{clean_ticker}_mae_info.pkl'
    
    # Check if all files exist
    if not all(os.path.exists(path) for path in [model_path, scaler_path, mae_info_path]):
        return None, None, None
    
    try:
        # Load MAE info first
        with open(mae_info_path, 'rb') as f:
            mae_info = pickle.load(f)
        
        # Check if MAE is <= 10%
        if mae_info['mae'] > 0.10:  # 10% threshold
            print(f"Saved model for {ticker} has MAE {mae_info['mae']:.4f} > 10%, will retrain")
            return None, None, None
        
        # Load model and scaler
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Loaded existing model for {ticker} with MAE: {mae_info['mae']:.4f}")
        return model, scaler, mae_info['mae']
        
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None, None, None

def evalModel(model, X_test, y_test, y_train):
    """Evaluate model and return metrics"""
    # Convert X_test and y_test to Numpy arrays if they are not already
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Ensure X_test is reshaped similarly to how X_train was reshaped
    # This depends on how you preprocessed the training data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Now evaluate the model on the test data
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss: ", test_loss)

    y_pred = model.predict(X_test, verbose=0)
    
    # Handle different prediction shapes
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.flatten()
    
    # Ensure both arrays have the same length
    min_length = min(len(y_test_flat), len(y_pred_flat))
    y_test_flat = y_test_flat[:min_length]
    y_pred_flat = y_pred_flat[:min_length]

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = mean_squared_error(y_test_flat, y_pred_flat)
    smape_value = smape(y_test_flat, y_pred_flat)
    mase_value = mase(y_test_flat, y_pred_flat, y_train)
    mape = mean_absolute_percentage_error(y_test_flat, y_pred_flat)

    print("Mean Absolute Error: ", mae)
    print("Root Mean Square Error: ", rmse)
    print("Symmetric Mean Absolute Percentage Error: ", smape_value)
    print("Mean Absolute Scaled Error: ", mase_value)
    print("Mean Absolute Percentage Error: ", mape)
    
    return mae, rmse, smape_value, mase_value, mape


# main check
if __name__ == "__main__":
    # List of tickers to process
    tickers = ['USDC-EUR', 'MXN=X', '^MXX', 'BTC-USD', 'ETH-USD', 'PAXG-USD', '^IXIC', '^SP500-45']
    # tickers = ['^IXIC']  # Uncomment for testing with single ticker
    
    print(f"\n🚀 Starting LSTM Forecast with Rate Limiting Protection")
    print(f"📊 Processing {len(tickers)} tickers with delays between requests")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Track processing statistics
    successful_tickers = []
    failed_tickers = []
    start_time = time.time()
    
    # Process each ticker one at a time with delays
    for index, ticker in enumerate(tickers):
        try:
            # Add delay between tickers to avoid rate limiting
            if index > 0:
                inter_ticker_delay = random.uniform(10, 20)  # 10-20 seconds between tickers
                print(f"\n⏳ Waiting {inter_ticker_delay:.1f} seconds before processing next ticker...")
                time.sleep(inter_ticker_delay)
            
            # Process the ticker
            success = process_single_ticker(ticker, index, len(tickers))
            
            if success:
                successful_tickers.append(ticker)
                print(f"✅ {ticker} completed successfully")
            else:
                failed_tickers.append(ticker)
                print(f"❌ {ticker} failed to process")
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Processing interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error with {ticker}: {e}")
            failed_tickers.append(ticker)
            continue
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"🏁 LSTM Forecast Processing Complete!")
    print(f"⏰ Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {len(successful_tickers)} tickers")
    print(f"❌ Failed: {len(failed_tickers)} tickers")
    
    if successful_tickers:
        print(f"✅ Successfully processed: {', '.join(successful_tickers)}")
    
    if failed_tickers:
        print(f"❌ Failed to process: {', '.join(failed_tickers)}")
    
    # Send summary to Telegram
    try:
        summary_message = f"🏁 <b>LSTM Forecast Batch Complete</b>\n\n"
        summary_message += f"⏰ <b>Total Time:</b> {total_time/60:.1f} minutes\n"
        summary_message += f"✅ <b>Successful:</b> {len(successful_tickers)}/{len(tickers)} tickers\n"
        summary_message += f"❌ <b>Failed:</b> {len(failed_tickers)}/{len(tickers)} tickers\n\n"
        
        if successful_tickers:
            summary_message += f"✅ <b>Processed:</b>\n"
            for ticker in successful_tickers:
                summary_message += f"• {ticker}\n"
        
        if failed_tickers:
            summary_message += f"\n❌ <b>Failed:</b>\n"
            for ticker in failed_tickers:
                summary_message += f"• {ticker}\n"
        
        summary_message += f"\n📊 All individual reports sent above ⬆️"
        
        asyncio.run(send_telegram(summary_message))
        print("📱 Summary sent to Telegram")
        
    except Exception as e:
        print(f"❌ Error sending summary to Telegram: {e}")
    
    print(f"🎉 Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")







