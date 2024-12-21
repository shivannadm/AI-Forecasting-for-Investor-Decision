import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
import math
from sklearn.metrics import mean_squared_error

def load_and_prepare_data(filepath):
    # Load the dataset
    filepath = "dataset\TATA.csv"
    df = pd.read_csv(filepath)

    # Extract necessary columns
    dates = df['Date']  # Use the Date column as is
    close_prices = df['Close']

    # Plot the original data with dates on x-axis
    plt.figure(figsize=(12, 6))
    plt.plot(dates, close_prices, label="Actual Prices")
    plt.title("Stock Price Data")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45, ticks=np.arange(0, len(dates), step=len(dates) // 10))
    plt.legend()
    plt.show()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_prices = scaler.fit_transform(np.array(close_prices).reshape(-1, 1))

    return dates, normalized_prices, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def build_and_train_model(X_train, y_train, X_test, y_test):
    # Build the CNN-LSTM model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

    return model

def predict_next_30_days(model, scaler, test_data, time_step=100):
    # Predict the next 30 days
    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []

    for i in range(30):
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:]).reshape((1, time_step, 1))
        else:
            x_input = np.array(temp_input).reshape((1, time_step, 1))

        yhat = model.predict(x_input, verbose=0)
        lst_output.extend(yhat.tolist())
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]

    # Transform predictions back to the original scale
    future_predictions = scaler.inverse_transform(lst_output)
    return future_predictions

def plot_results(dates, normalized_prices, scaler, train_predict, test_predict, time_step):
    # Transform predictions back to original scale
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Plot actual vs predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates, scaler.inverse_transform(normalized_prices), label="Actual Stock Price")
    plt.plot(dates[time_step:len(train_predict) + time_step], train_predict, label="Train Prediction")
    plt.plot(dates[len(train_predict) + (time_step * 2) + 1:len(train_predict) + (time_step * 2) + 1 + len(test_predict)], test_predict, label="Test Prediction")
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.xticks(rotation=45, ticks=np.arange(0, len(dates), step=len(dates) // 10))
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    filepath = "data/dataset.csv"  # Replace with the path to your dataset
    time_step = 100

    dates, normalized_prices, scaler = load_and_prepare_data(filepath)

    # Prepare training and testing datasets
    training_size = int(len(normalized_prices) * 0.65)
    train_data = normalized_prices[0:training_size, :]
    test_data = normalized_prices[training_size:len(normalized_prices), :]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input data into [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train the model
    model = build_and_train_model(X_train, y_train, X_test, y_test)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Plot results
    plot_results(dates, normalized_prices, scaler, train_predict, test_predict, time_step)

    # Predict the next 30 days
    future_predictions = predict_next_30_days(model, scaler, test_data, time_step)
    print("Future Predictions:", future_predictions)
