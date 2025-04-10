import pandas as pd
import numpy as np
import tensorflow
import keras
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, AdditiveAttention, Permute, Reshape, Multiply, Attention, Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import telegram
import asyncio



# main check
if __name__ == "__main__":

    for Ticker in ['USDC-EUR', 'MXN=X', '^MXX', 'BTC-USD', 'ETH-USD', 'PAXG-USD', '^IXIC', '^SP500-45']:
    # for Ticker in ['^IXIC']:
        # Download the data1
        data = yf.download(Ticker, period='6y', interval='1d', timeout=20)
        # data.to_csv(f'data/{Ticker}_tickers.csv')
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        mae = None
        rmse = None
        smape_value = None
        mase_value = None

        async def send_telegram(message):
            # Telegram Bot Token (Replace with your own bot token)
            BOT_TOKEN = '6774919345:AAEdbuOoggOwp8IqfuQky1tL97i4HXBFY-M'

            # Chat ID (Replace with your own chat ID)
            CHAT_ID = '-1002687756429' #'955478477'
            # CHAT_ID = '955478477'

            # Initialize the Telegram bot
            bot = telegram.Bot(token=BOT_TOKEN)

            try:
                await bot.send_message(chat_id=CHAT_ID, text=message)
                print("Telegram message sent successfully.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
            return True


        async def send_image_to_telegram(image_path, caption=None):
            """
            Sends an image to a Telegram chat.

            Args:
                bot_token: The API token of your Telegram bot.
                chat_id: The ID of the chat to send the image to.
                image_path: The file path of the image to send.
                caption: (Optional) A caption to include with the image.
            """

            bot_token = '6774919345:AAEdbuOoggOwp8IqfuQky1tL97i4HXBFY-M'
            chat_id = '-1002687756429' #'955478477'
            # chat_id = '955478477'
            try:
                bot = telegram.Bot(token=bot_token)
                async with bot:
                    if caption:
                        await bot.send_photo(chat_id=chat_id, photo=open(image_path, 'rb'), caption=caption)
                    else:
                        await bot.send_photo(chat_id=chat_id, photo=open(image_path, 'rb'))
                print("Image sent successfully!")

            except Exception as e:
                print(f"Error sending image: {e}")

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
            return np.mean(np.abs((X - y) / ((np.abs(X) + np.abs(y)) / 2))) * 100

        def mase(y_true, y_pred, y_train):
            # Calculate the mean absolute error of the training data
            mae_train = np.mean(np.abs(y_train - np.mean(y_train)))
            # Calculate the mean absolute error of the test data
            mae_test = np.mean(np.abs(y_true - y_pred))
            # Calculate the MASE
            return mae_test / mae_train

        def evalModel(model, X_test, y_test):
            global mae, rmse, mase_value, smape_value
            # Convert X_test and y_test to Numpy arrays if they are not already
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            # Ensure X_test is reshaped similarly to how X_train was reshaped
            # This depends on how you preprocessed the training data
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Now evaluate the model on the test data
            test_loss = model.evaluate(X_test, y_test)
            print("Test Loss: ", test_loss)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred)
            smape_value = smape(y_test, y_pred)
            mase_value = mase(y_test, y_pred, y_train)

            print("Mean Absolute Error: ", mae)
            print("Root Mean Square Error: ", rmse)
            print("Symmetric Mean Absolute Percentage Error: ", smape_value)
            print("Mean Absolute Scaled Error: ", mase_value)


        X, y = create_dataset(scaled_data)
        X_train, y_train, X_test, y_test = split_data(X, y)

        # Build the LSTM model
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

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
        model.compile(optimizer='adam', loss='mean_squared_error')

        # train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        print(model.summary())

        evalModel(model, X_test, y_test)

        # Fetching the latest 60 days of AAPL stock data
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
            next_prediction = model.predict(current_batch)

            # Reshape the prediction to fit the batch dimension
            next_prediction_reshaped = next_prediction.reshape(1, 1, 1)

            # Append the prediction to the batch used for predicting
            current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

            # Inverse transform the prediction to the original price scale
            predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

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
        image_name = f'{Ticker}_predictions.png'
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
        if Ticker == '^MXN=X':
            Ticker = 'USD/MXN'
        if Ticker == '^SP500-45':
            Ticker = 'S&P 500 - Information Technology'

        plt.title(f"{Ticker} Stock Price: Last 60 Days and Next 4 Days Predicted")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        image_name_full = f'full {Ticker}_predictions.png'
        plt.savefig(image_name_full)
        plt.close()
        # plt.show()

        asyncio.run(send_telegram(f'Here are the next 10 days predictions for {Ticker} stock prices.'))
        asyncio.run(send_telegram(f'Predicted Stock Prices for the next 10 days: {predicted_prices}'))
        asyncio.run(send_telegram(f'Mean Absolute Error: % {mae*100:.2f}'))
        asyncio.run(send_telegram(f'Mean Absolute Scaled Error: % {mase_value*100:.2f}'))
        asyncio.run(send_telegram(f'Symmetric Mean Absolute Percentage Error: % {smape_value*100:.2f}'))
        asyncio.run(send_telegram(f'Root Mean Square Error: % {rmse*100:.2f}'))
        asyncio.run(send_telegram(f'Here is the plot:'))
        asyncio.run(send_image_to_telegram(image_name, caption='Predicted Stock Prices for the next 10 days'))
        asyncio.run(send_image_to_telegram(image_name_full, caption='Predicted Stock Prices for the next 10 days'))







