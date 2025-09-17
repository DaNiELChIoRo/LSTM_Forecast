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
from keras.models import load_model


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

    for Ticker in ['USDC-EUR', 'MXN=X', '^MXX', 'BTC-USD', 'ETH-USD', 'PAXG-USD', '^IXIC', '^SP500-45']:
    # for Ticker in ['^IXIC']:
        try:
            print(f"Processing {Ticker}...")
            # Download the data1
            data = yf.download(Ticker, period='6y', interval='1d', timeout=20)
            
            # Check if data is empty
            if data.empty or len(data) < 61:  # Need at least 61 days for 60-day lookback
                print(f"Insufficient data for {Ticker}, skipping...")
                continue
                
            print(f"Downloaded {len(data)} days of data for {Ticker}")
            
            # data.to_csv(f'data/{Ticker}_tickers.csv')
            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
        except Exception as e:
            print(f"Error downloading data for {Ticker}: {e}")
            continue

        mae = None
        rmse = None
        smape_value = None
        mase_value = None
        mape = None

        X, y = create_dataset(scaled_data)
        X_train, y_train, X_test, y_test = split_data(X, y)

        # Try to load existing model first
        model, loaded_scaler, loaded_mae = load_model_and_scaler(Ticker)
        
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
                save_model_and_scaler(model, scaler, Ticker, mae)
            else:
                print(f"Model MAE {mae:.4f} > 10%, not saving model")

        # Analyze market sentiment
        print("Analyzing market sentiment...")
        try:
            sentiment_data = get_ticker_sentiment(Ticker)
            sentiment_message = format_sentiment_for_telegram(Ticker, sentiment_data)
            print(f"Sentiment Analysis for {Ticker}:")
            print(f"  Score: {sentiment_data['sentiment_score']:.3f}")
            print(f"  Label: {sentiment_data['sentiment_label']}")
            print(f"  Confidence: {sentiment_data['confidence']*100:.0f}%")
        except Exception as e:
            print(f"Error analyzing sentiment for {Ticker}: {e}")
            sentiment_message = f"\n📊 <b>Sentiment Analysis for {Ticker}</b>\n❌ Could not analyze sentiment\n"

        # Fetching the latest 60 days of stock data
        data = data.iloc[-60:]  # Get the last 60 days of data
        # yf.download(Ticker, period='61d', interval='1d')

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
        plt.title(f'{Ticker} Stock Price Prediction')
        plt.xticks(rotation=90)

        # Save plot as image to send it to the user
        # Clean ticker name for file naming
        clean_ticker = Ticker.replace('/', '_').replace('^', '').replace('=', '_')
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
        if Ticker == '^IXIC':
            Ticker = 'NASDAQ Composite'
        if Ticker == '^MXX':
            Ticker = 'IPC MEXICO'
        if Ticker == 'MXN=X':
            Ticker = 'USD/MXN'
        if Ticker == '^SP500-45':
            Ticker = 'S&P 500 - Information Technology'

        plt.title(f"{Ticker} Stock Price: Last 60 Days and Next 4 Days Predicted")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        image_name_full = f'full {clean_ticker}_predictions.png'
        plt.savefig(image_name_full)
        plt.close()
        # plt.show()

        try:
            # Create comprehensive message with predictions and sentiment
            prediction_message = f'🔮 <b>LSTM Forecast Report for {Ticker}</b>\n\n'
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
            asyncio.run(send_image_to_telegram(image_name, caption=f'📊 {Ticker} - Next 10 Days Prediction'))
            asyncio.run(send_image_to_telegram(image_name_full, caption=f'📈 {Ticker} - Historical Data + Predictions'))
        except Exception as e:
            print(f"Error sending Telegram message for {Ticker}: {e}")
            
        print(f"✅ Successfully processed {Ticker}")
        print("-" * 50)







