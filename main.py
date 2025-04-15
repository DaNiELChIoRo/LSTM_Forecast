import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, AdditiveAttention, Permute, Reshape, Multiply, Attention, Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from telegram_sender import send_telegram, send_image_to_telegram
import asyncio



# main check
if __name__ == "__main__":
    best_mae = None
    best_batch_size = None
    best_epoch = None
    best_activation_function = None
    best_loss_function = None
    best_mase_value = None
    best_smape_value = None
    best_rmse_value = None

    # for Ticker in ['USDC-EUR', 'MXN=X', '^MXX', 'BTC-USD', 'ETH-USD', 'PAXG-USD', '^IXIC', '^SP500-45']:
    for Ticker in ['PAXG-USD']:
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
        for loss_func in ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']:
            for activation_func in ['relu', 'sigmoid', 'tanh', 'softmax', 'linear']:
                for batch_size in [16, 32, 64]:
                    for epoch in [25, 50, 75, 100]:
                        # Build the LSTM model
                        model = Sequential()

                        model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))))
                        model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1))))
                        model.add(Bidirectional(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1))))
                        model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1))))
                        model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))))

                        model.add(Dropout(0.2))
                        model.add(Activation(activation_func))
                        model.add(BatchNormalization())
                        model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
                        # The attention mechanism
                        attention = AdditiveAttention(name="attention_weight")

                        model.add(Flatten())

                        # Final Dense Layer
                        model.add(Dense(units=1))


                        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
                        # compile the model
                        model.compile(optimizer='adam', loss=loss_func)

                        # train the model
                        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

                        evalModel(model, X_test, y_test)

                        if best_mae is None or mae < best_mae:
                            best_mae = mae
                            best_batch_size = batch_size
                            best_epoch = epoch
                            best_activation_function = activation_func
                            best_loss_function = loss_func
                            best_mase_value  = mase_value
                            best_smape_value = smape_value
                            best_rmse_value  = rmse

                        del model



        # asyncio.run(send_telegram(f'Here are the next 10 days predictions for {Ticker} stock prices.'))
        # asyncio.run(send_telegram(f'Predicted Stock Prices for the next 10 days: {predicted_prices}'))
        asyncio.run(send_telegram(f'Best Mean Absolute Error: %<b>{best_mae*100:.2f}</b>\n\
        Best Mean Absolute Scaled Error: %<b>{best_mase_value*100:.2f}</b>\n\
        Best Symmetric Mean Absolute Percentage Error: %<b>{best_smape_value*100:.2f}</b>\n\
        Best Root Mean Square Error: %<b>{best_rmse_value*100:.2f}</b>\n\
        Best Batch Size: <b>{best_batch_size}</b>\n\
        Best Epoch: <b>{best_epoch}</b>\n\
        Best Activation Function: <b>{best_activation_function}</b>\n\
        Best Loss Function: <b>{best_loss_function}</b>\n\
        '))
        # asyncio.run(send_telegram(f'Here is the plot:'))
        # asyncio.run(send_image_to_telegram(image_name, caption='Predicted Stock Prices for the next 10 days'))
        # asyncio.run(send_image_to_telegram(image_name_full, caption='Predicted Stock Prices for the next 10 days'))







