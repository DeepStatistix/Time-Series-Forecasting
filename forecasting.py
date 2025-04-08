# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Apple Data.csv')

df

df = df.rename(columns={
    'state_name': 'State',
    'district_name': 'District',
    'market_name': 'Market',
    'date_arrival': 'Date',
    'Arrival': 'Arrival (Tonnes)',
    'MIN': 'Minimum Price (Rs./Quintal)',
    'MAX': 'Maximum Price (Rs./Quintal)',
    'MODAL': 'Modal Price (Rs./Quintal)',
    'Month': 'Month Number',
    'Year': 'Year Number',
    'Month_name': 'Month Name'
})

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

df['Market'].unique()

bangalore_data = df[df['Market'] == 'Binny Mill (F&V), Bangalore']
plt.figure(figsize=(12, 6), facecolor= 'white')
plt.plot(bangalore_data.index, bangalore_data['Modal Price (Rs./Quintal)'], marker='o', linestyle='-', color='blue', label='Modal Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Modal Price Trend in Parimpore Market')
plt.legend()  # Show legend with label specified in the plot
plt.grid(True)  # Show gridlines for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

from statsmodels.tsa.stattools import adfuller
time_series_data = df[df['Market'] == 'Binny Mill (F&V), Bangalore']['Modal Price (Rs./Quintal)']
# Assuming your time series data is stored in a variable named 'time_series_data'
result_adf = adfuller(time_series_data)
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
print('Critical Values:', result_adf[4])

from statsmodels.tsa.stattools import kpss

# Assuming your time series data is stored in a variable named 'time_series_data'
result_kpss = kpss(time_series_data)
print('KPSS Statistic:', result_kpss[0])
print('p-value:', result_kpss[1])
print('Critical Values:', result_kpss[3])

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.figure(figsize=(12,6))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

test_stationarity(bangalore_data['Modal Price (Rs./Quintal)'])

result = seasonal_decompose(bangalore_data['Modal Price (Rs./Quintal)'], model='multiplicative', period = 52)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)

bangaloreprice = df[df['Market'] == 'Binny Mill (F&V), Bangalore'][['Date', 'Modal Price (Rs./Quintal)']]

bangaloreprice['Date'] = pd.to_datetime(bangaloreprice['Date'])

bangaloreprice.set_index('Date',inplace=True)

bangaloreprice

train_size = int(len(bangaloreprice) * 0.9)
train_data = bangaloreprice.iloc[:train_size]
test_data = bangaloreprice.iloc[train_size:]

# Plotting the train and test data with enhanced styling
plt.figure(figsize=(12, 4))

# Plotting train data with a solid green line
plt.plot(train_data.index, train_data['Modal Price (Rs./Quintal)'], color='green', linestyle='-', marker='o', markersize=5, label='Train data')

# Plotting test data with a solid blue line
plt.plot(test_data.index, test_data['Modal Price (Rs./Quintal)'], color='blue', linestyle='-', marker='o', markersize=5, label='Test data')

plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.title('Train and Test Data')
plt.legend()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Assuming your time series data is stored in 'data'
# plot_acf
plt.figure(figsize=(10, 4))
plot_acf(bangaloreprice, lags=20, alpha=0.05, ax=plt.gca())
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.grid(True)
plt.show()

# plot_pacf
plt.figure(figsize=(10, 4))
plot_pacf(bangaloreprice, lags=20, alpha=0.05, ax=plt.gca())
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.grid(True)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Assuming your time series data is stored in 'data'
# plot_acf
plt.figure(figsize=(12, 6))
plot_acf(train_data['Modal Price (Rs./Quintal)'], lags=50, alpha=0.05, ax=plt.gca())
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.grid(True)
plt.show()

# plot_pacf
plt.figure(figsize=(12, 6))
plot_pacf(train_data['Modal Price (Rs./Quintal)'], lags=50, alpha=0.05, ax=plt.gca())
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.grid(True)
plt.show()

pip install pmdarima

import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from pmdarima import auto_arima

max_p = 3  # Maximum autoregression order
max_d = 2  # Maximum differencing order
max_q = 3  # Maximum moving average order

# Perform AutoARIMA with specified maximum orders
model = auto_arima(train_data, start_p=0, d=0, start_q=0,
                    max_p=max_p, max_d=max_d, max_q=max_q,
                    seasonal=False,  # Assuming non-seasonal data
                    trace=True,  # Set to True to see the stepwise search
                    error_action='ignore',  # Set to 'ignore' to avoid errors that can arise from certain parameter combinations
                    suppress_warnings=True)  # Suppress warnings

fitted = model.fit(train_data)
print(fitted.summary())

test_predictions = fitted.predict(len(test_data))

# Print or use the predicted values for the test data
print(test_predictions)

#forecast_values= fitted.predict(steps=len(test_data))

# Extract forecasted values and calculate confidence intervals
#forecast = forecast_values
# Plotting actual test data and predicted values
plt.figure(figsize=(12, 4))
plt.plot(train_data, label='Train Data')
plt.plot(test_data, label='Test Data')
plt.plot(test_data.index,test_predictions, color='green', label='Predicted')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

mse = mean_squared_error(test_data['Modal Price (Rs./Quintal)'], test_predictions)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data['Modal Price (Rs./Quintal)'], test_predictions)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data['Modal Price (Rs./Quintal)'], test_predictions))
print('RMSE: '+str(rmse))

plt.figure(figsize=(10,3))
plt.plot(test_data['Modal Price (Rs./Quintal)'].values, color = 'red')
plt.plot(test_predictions.values, color = 'blue')
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

predicted = pd.DataFrame({
    'Actual': test_data['Modal Price (Rs./Quintal)'].values,
    'Predicted': test_predictions.values
}, index=test_data.index)

predicted

predicted.to_csv('ARIMApredicted.csv')

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
print('MAPE for training data: ', mape(test_data['Modal Price (Rs./Quintal)'].values,predictions.values))

import statsmodels.api as sm
model = sm.tsa.ARIMA(train_data, order=(2,1,1))
fitted_model = model.fit()

fitted_model.summary()

forecast = fitted_model.get_forecast(steps=len(test_data))
predicted_mean = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Option 2: Use predict to get predictions without confidence intervals
# Specify dynamic=False to predict one-step ahead values
start = len(train_data)
end = len(train_data) + len(test_data) - 1
predictions = fitted_model.predict(start=start, end=end, dynamic=False)

# Step 4: Plot the results to compare predictions with actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, predictions, color='red', label='Predicted')
plt.title('Test Data vs. Predicted')
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(train_data, label='Train Data')
plt.plot(test_data, label='Test Data')
plt.plot(test_data.index,predictions, color='green', label='Predicted')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

plt.figure(figsize=(10,3))
plt.plot(test_data['Modal Price (Rs./Quintal)'].values, color = 'red')
plt.plot(predictions.values, color = 'blue')
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

mse = mean_squared_error(test_data['Modal Price (Rs./Quintal)'], predictions)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data['Modal Price (Rs./Quintal)'], predictions)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data['Modal Price (Rs./Quintal)'], predictions))
print('RMSE: '+str(rmse))

predicted = pd.DataFrame({
    'Actual': test_data['Modal Price (Rs./Quintal)'].values,
    'Predicted': predictions.values
}, index=test_data.index)

predicted

predicted.to_csv('ARIMApredicted.csv')

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
print('MAPE for training data: ', mape(test_data['Modal Price (Rs./Quintal)'].values,predictions.values))

fitted_model.plot_diagnostics(figsize=(15, 8))
plt.show()

import statsmodels.api as sm
model = sm.tsa.ARIMA(bangaloreprice, order=(2,1,1))
fitted_model = model.fit()
# Make predictions for the future 52 weeks along with confidence intervals
forecast_values = fitted_model.get_forecast(steps=21)
forecast = forecast_values.predicted_mean
confidence_intervals = forecast_values.conf_int()

last_date_str = '2022-12-25'  # Example date string in 'YYYY-MM-DD' format
last_date = pd.to_datetime(last_date_str)  # Convert string to pandas datetime

# Then perform the addition with Timedelta
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=21, freq='W')

# Plotting original data
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index, bangaloreprice['Modal Price (Rs./Quintal)'], label='Original Data', color='blue')

# Plotting forecasted data and confidence intervals
plt.plot(future_dates, forecast, label='Forecast', color='red')
plt.fill_between(future_dates, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')

# Formatting
plt.title('Forecasted Price with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

"""# SVM"""

from sklearn.model_selection import TimeSeriesSplit
def evaluate_timesteps(y, timesteps_list):
    results = {}
    for timesteps in timesteps_list:
        if len(y) < timesteps:
            print(f"Not enough data points for timesteps = {timesteps}")
            continue

        # Create sequences
        data_timesteps = np.array([y[i:i + timesteps] for i in range(len(y) - timesteps)])

        # Split into features and target
        X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        mse_list = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_seq[train_index], y_seq[test_index]

            # Scale the features
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            # Fit the SVM model with different parameters
            model = SVR(kernel='rbf', gamma='scale', C=1, epsilon=0.01)
            model.fit(X_train_scaled, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        results[timesteps] = avg_mse

    return results

# Example usage
y = bangaloreprice.values.flatten()
timesteps_list = [10, 15, 20, 25, 30, 35]
results = evaluate_timesteps(y, timesteps_list)
print(results)

from sklearn.preprocessing import MinMaxScaler

y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 15
# Create sequences
data_timesteps = np.array([y_scaled[i:i + timesteps].flatten() for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.03, 0.1, 0.3],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly']
}

# Initialize the model
svr = SVR()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

from sklearn.svm import SVR
model = SVR(kernel='linear',gamma='scale', C=1, epsilon = 0.01)

model.fit(X_train, y_train)

support_vectors = model.support_vectors_
print(f"Number of Support Vectors: {len(support_vectors)}")

y_train_pred = model.predict(X_train).reshape(-1,1)
y_test_pred = model.predict(X_test).reshape(-1,1)

# Reshape y_train and y_test to 2D arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# Inverse transform
y_train_original = scaler.inverse_transform(y_train)
y_test_original = scaler.inverse_transform(y_test)

y_train_pred = scaler.inverse_transform(y_train_pred)

y_test_pred = scaler.inverse_transform(y_test_pred)

plt.figure(figsize=(15,6))
plt.plot(y_train_original, color = 'red')
plt.plot(y_train_pred, color = 'blue')
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

plt.figure(figsize=(10,3))
plt.plot(y_test_original, color = 'red')
plt.plot(y_test_pred, color = 'blue')
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
# Absolute metrics
mse = mean_squared_error(y_test_original, y_test_pred)
mae = mean_absolute_error(y_test_original, y_test_pred)
rmse = np.sqrt(mse)

# Relative metrics
mape = mape(y_test_pred,y_test_original)
smape_value = smape(y_test_original, y_test_pred)

# Directional accuracy
correct_directions = np.sign(y_test_original[1:] - y_test_original[:-1]) == np.sign(y_test_pred[1:] - y_test_pred[:-1])
directional_accuracy = np.mean(correct_directions)

# Print results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'SMAPE: {smape_value:.2f}%')
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

y_test_original_flat = y_test_original.flatten()
y_test_pred_flat = y_test_pred.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
}, index=test_data.index[:-1])

predicted

predicted.to_csv('svmpredicted.csv')

y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 15
# Create sequences
data_timesteps = np.array([[y_scaled[i+j] for j in range(timesteps)] for i in range(len(y_scaled)-timesteps+1)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Flatten X
X = X.reshape(X.shape[0], -1)

# Train SVR model on the entire dataset
model = SVR(kernel='linear',gamma='scale', C=1, epsilon = 0.01)
model.fit(X, y)

last_sequence = X[-1]  # Use the last sequence from the data
future_predictions = []

future_weeks = 26
for i in range(future_weeks):
    # Predict using the SVR model
    pred = model.predict(last_sequence.reshape(1, -1))
    future_predictions.append(pred[0])

    # Update the last sequence for the next prediction
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = pred

# Inverse transform the predictions to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future timestamps for the next 52 weeks
last_date = bangaloreprice.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=future_weeks, freq='W')

# Plotting the forecast for future dates
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index, bangaloreprice.values, label='Original Data', color='red')
plt.plot(future_dates, future_predictions, label='SVR Forecast for Future', color='blue')
plt.legend()
plt.title('Forecasting for Next 52 Weeks using SVR')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def evaluate_timesteps(y, timesteps_list, scaler):
    results = {}

    for timesteps in timesteps_list:
        # Create sequences for the current timestep
        data_timesteps = np.array([[y[i + j] for j in range(timesteps)] for i in range(len(y) - timesteps + 1)])

        # Split into features and target
        X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

        # Reshape input data to be 3D (samples, timesteps, features) for RNN
        X = X.reshape(X.shape[0], timesteps - 1, 1)

        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        mse_list = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_seq[train_index], y_seq[test_index]

            # Build Simple RNN model
            model = Sequential()
            model.add(SimpleRNN(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Predict and inverse transform predictions
            y_pred = model.predict(X_test)
            y_pred_inverse = scaler.inverse_transform(y_pred)
            y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Calculate and store the MSE for the current fold
            mse = mean_squared_error(y_test_inverse, y_pred_inverse)
            mse_list.append(mse)

        # Store the average MSE for the current timestep
        avg_mse = np.mean(mse_list)
        results[timesteps] = avg_mse

    return results

# Define your timestep range
timesteps_list = [10, 15, 20, 30, 40, 50]
# Assume y_scaled is already scaled using the scaler
results = evaluate_timesteps(y_scaled, timesteps_list, scaler)

# Find the timestep with the lowest MSE
optimal_timesteps = min(results, key=results.get)
print(f"Optimal timesteps: {optimal_timesteps}, MSE: {results[optimal_timesteps]}")

def evaluate_timesteps(y, timesteps_list, scaler):
    results = {}

    for timesteps in timesteps_list:
        # Create sequences for the current timestep
        data_timesteps = np.array([[y[i + j] for j in range(timesteps)] for i in range(len(y) - timesteps + 1)])

        # Split into features and target
        X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

        # Reshape input data to be 3D (samples, timesteps, features) for RNN
        X = X.reshape(X.shape[0], timesteps - 1, 1)

        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        mse_list = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_seq[train_index], y_seq[test_index]

            # Build Simple RNN model
            model = Sequential()
            model.add(SimpleRNN(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Predict and inverse transform predictions
            y_pred = model.predict(X_test)
            y_pred_inverse = scaler.inverse_transform(y_pred)
            y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Calculate and store the MSE for the current fold
            mse = mean_squared_error(y_test_inverse, y_pred_inverse)
            mse_list.append(mse)

        # Store the average MSE for the current timestep
        avg_mse = np.mean(mse_list)
        results[timesteps] = avg_mse

    return results

# Split the data into training and testing sets
train_size = int(len(bangaloreprice) * 0.9)
train_data = bangaloreprice[:train_size]

# Convert the relevant column of train_data to a NumPy array
train_data_array = train_data['Modal Price (Rs./Quintal)'].values  # Replace 'column_name' with the actual column name

# Scale the training data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data_array.reshape(-1, 1)).reshape(-1)

# Example time series data
timesteps_list = [10, 15, 20, 30, 40, 50]
results = evaluate_timesteps(train_data_scaled, timesteps_list, scaler)
print("Optimal Timesteps:", min(results, key=results.get), "with MSE:", min(results.values()))

y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 15

# Create sequences
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Reshape input data to be 3D (samples, timesteps, features) for RNN
X = X.reshape(X.shape[0], timesteps -1, 1)

train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

test = bangaloreprice.index[train_size + timesteps:]

from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(SimpleRNN(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
                               verbose=1)

# Train the model with validation split, early stopping, and plot the history
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Get predictions on the training set
y_train_pred = model.predict(X_train)

# Plot the training data
plt.figure(figsize=(15, 6))
plt.plot(y_train, label='Actual Train Data', color='blue')

# Plot the predictions on the training set
plt.plot(range(len(y_train)), y_train_pred, label='Predicted Train Data', color='red')

plt.title('RNN Model - Training Data and Predictions')
plt.xlabel('Timestamp')
plt.legend()
plt.show()

# Make predictions on the test set
y_test_pred = model.predict(X_test)
# Inverse transform the predictions to original scale
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 3))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_test_pred, label='Predicted Prices', color='red')
plt.title('RNN Time Series Forecasting')
plt.xlabel('Timestamp')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_test_pred)
print('MSE: '+str(mse))
mae = mean_absolute_error(y_test, y_test_pred)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE: '+str(rmse))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
print('MAPE for test data: ', mape(y_test, y_test_pred))

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
# Absolute metrics
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

# Relative metrics
mape = mape(y_test_pred,y_test)
smape_value = smape(y_test, y_test_pred)

# Directional accuracy
correct_directions = np.sign(y_test[1:] - y_test[:-1]) == np.sign(y_test_pred[1:] - y_test_pred[:-1])
directional_accuracy = np.mean(correct_directions)

# Print results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'SMAPE: {smape_value:.2f}%')
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

y_test_original_flat = y_test.flatten()
y_test_pred_flat = y_test_pred.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
}, index=test)

predicted

predicted.to_csv('rnn_predict.csv')

# Define your timesteps
timesteps = 20  # Adjust as needed

# Load and prepare the data
y = bangaloreprice.values.flatten()  # Ensure y is a 1D array

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Create sequences
data_timesteps = np.array([y_scaled[i:i + timesteps] for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

# Build RNN model
model = Sequential()
model.add(SimpleRNN(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='loss',
                               patience=10,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
                               verbose=1)

# Train the model on the whole dataset
history = model.fit(X, y_seq,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping])

# Prepare the last sequence from the entire data for forecasting
last_sequence = np.array(y_scaled[-(timesteps - 1):]).reshape(1, timesteps - 1, 1)

# Forecast future values for 'future_weeks'
future_weeks = 52
future_timestamps = pd.date_range(start=bangaloreprice.index[-1] + pd.DateOffset(weeks=1),
                                  periods=future_weeks,
                                  freq='W')

future_predictions = []
for _ in range(future_weeks):
    # Predict the next value
    pred = model.predict(last_sequence)[0, 0]
    future_predictions.append(pred)

    # Update the sequence for the next prediction
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = pred

# Convert predictions to a numpy array and reshape for inverse transform
future_predictions = np.array(future_predictions).reshape(-1, 1)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(future_predictions)

# Plotting the forecast for future dates
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index, scaler.inverse_transform(y_scaled), label='Actual Prices', color='blue')
plt.plot(future_timestamps, future_predictions, label='Forecast for Future', color='red')
plt.title('RNN Time Series Forecasting - Future Forecast')
plt.xlabel('Date')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.legend()
plt.show()

def evaluate_timesteps(y, timesteps_list, scaler):
    results = {}

    for timesteps in timesteps_list:
        # Create sequences for the current timestep
        data_timesteps = np.array([[y[i + j] for j in range(timesteps)] for i in range(len(y) - timesteps + 1)])

        # Split into features and target
        X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

        # Reshape input data to be 3D (samples, timesteps, features) for RNN
        X = X.reshape(X.shape[0], timesteps - 1, 1)

        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        mse_list = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_seq[train_index], y_seq[test_index]

            # Build Simple RNN model
            model = Sequential()
            model.add(LSTM(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Predict and inverse transform predictions
            y_pred = model.predict(X_test)
            y_pred_inverse = scaler.inverse_transform(y_pred)
            y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Calculate and store the MSE for the current fold
            mse = mean_squared_error(y_test_inverse, y_pred_inverse)
            mse_list.append(mse)

        # Store the average MSE for the current timestep
        avg_mse = np.mean(mse_list)
        results[timesteps] = avg_mse

    return results

# Split the data into training and testing sets
train_size = int(len(bangaloreprice) * 0.9)
train_data = bangaloreprice[:train_size]

# Convert the relevant column of train_data to a NumPy array
train_data_array = train_data['Modal Price (Rs./Quintal)'].values  # Replace 'column_name' with the actual column name

# Scale the training data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data_array.reshape(-1, 1)).reshape(-1)

# Example time series data
timesteps_list = [10, 15, 20, 30, 40, 50]
results = evaluate_timesteps(train_data_scaled, timesteps_list, scaler)
print("Optimal Timesteps:", min(results, key=results.get), "with MSE:", min(results.values()))

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def evaluate_timesteps(y, timesteps_list, scaler):
    results = {}

    for timesteps in timesteps_list:
        # Create sequences for the current timestep
        data_timesteps = np.array([[y[i + j] for j in range(timesteps)] for i in range(len(y) - timesteps + 1)])

        # Split into features and target
        X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

        # Reshape input data to be 3D (samples, timesteps, features) for RNN
        X = X.reshape(X.shape[0], timesteps - 1, 1)

        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        mse_list = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_seq[train_index], y_seq[test_index]

            # Build Simple RNN model
            model = Sequential()
            model.add(LSTM(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Predict and inverse transform predictions
            y_pred = model.predict(X_test)
            y_pred_inverse = scaler.inverse_transform(y_pred)
            y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Calculate and store the MSE for the current fold
            mse = mean_squared_error(y_test_inverse, y_pred_inverse)
            mse_list.append(mse)

        # Store the average MSE for the current timestep
        avg_mse = np.mean(mse_list)
        results[timesteps] = avg_mse

    return results

# Define your timestep range
timesteps_list = [10, 15, 20, 30, 40, 50]
# Assume y_scaled is already scaled using the scaler
results = evaluate_timesteps(y_scaled, timesteps_list, scaler)

# Find the timestep with the lowest MSE
optimal_timesteps = min(results, key=results.get)
print(f"Optimal timesteps: {optimal_timesteps}, MSE: {results[optimal_timesteps]}")

y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 10
# Create sequences
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Reshape input data to be 3D (samples, timesteps, features) for RNN
X = X.reshape(X.shape[0], timesteps -1, 1)

train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

from keras.layers import Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Build LSTM model with regularization
model_lstm = Sequential()
model_lstm.add(LSTM(units=100, activation='relu', input_shape=(timesteps-1, 1)))
model_lstm.add(Dense(1))
#model_lstm.add(Dropout(0.1))  # Adjust the dropout rate as needed
# model_lstm.add(Dense(units=1, kernel_regularizer=l2(0.03)))  # L2 regularization added
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model with validation split and early stopping
history = model_lstm.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get predictions on the training set
y_train_pred = model_lstm.predict(X_train)

# Plot the training data
plt.figure(figsize=(15, 6))
plt.plot(y_train, label='Actual Train Data', color='blue')

# Plot the predictions on the training set
plt.plot(range(len(y_train)), y_train_pred, label='Predicted Train Data', color='red')

plt.title('LSTM Model - Training Data and Predictions')
plt.xlabel('Timestamp')
plt.legend()
plt.show()

# Predictions on test data
y_test_pred_lstm = model_lstm.predict(X_test)
y_test_pred_lstm = scaler.inverse_transform(y_test_pred_lstm)

# Plotting the predictions on the test set
plt.figure(figsize=(10, 3))
plt.plot(scaler.inverse_transform(y_test), label='Actual Prices', color='blue')
plt.plot(y_test_pred_lstm, label='Predicted Prices LSTM', color='red')
plt.title('LSTM Time Series Forecasting - Test Set')
plt.xlabel('Timestamp')
plt.legend()
plt.show()

mse = mean_squared_error(scaler.inverse_transform(y_test), y_test_pred_lstm)
print('MSE: '+str(mse))
mae = mean_absolute_error(scaler.inverse_transform(y_test), y_test_pred_lstm)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test), y_test_pred_lstm))
print('RMSE: '+str(rmse))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
print('MAPE for test data: ', mape(scaler.inverse_transform(y_test), y_test_pred_lstm))

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
# Absolute metrics
mse = mean_squared_error(scaler.inverse_transform(y_test), y_test_pred_lstm)
mae = mean_absolute_error(scaler.inverse_transform(y_test), y_test_pred_lstm)
rmse = np.sqrt(mse)

# Relative metrics
mape = mape(y_test_pred_lstm,scaler.inverse_transform(y_test))
smape_value = smape(scaler.inverse_transform(y_test), y_test_pred_lstm)

# Directional accuracy
correct_directions = np.sign(scaler.inverse_transform(y_test)[1:] - scaler.inverse_transform(y_test)[:-1]) == np.sign(y_test_pred_lstm[1:] - y_test_pred_lstm[:-1])
directional_accuracy = np.mean(correct_directions)

# Print results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'SMAPE: {smape_value:.2f}%')
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

test = bangaloreprice.index[train_size + timesteps:]

y_test = scaler.inverse_transform(y_test)
y_test_original_flat = y_test.flatten()
y_test_pred_flat = y_test_pred_lstm.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
}, index=test)

predicted

predicted.to_csv('lstm_predicted.csv')

timesteps = 50

# Prepare data with 50 timesteps
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

data_sequences = create_sequences(y, timesteps)
X, y_seq = data_sequences[:, :-1], data_sequences[:, -1]
X = X.reshape(X.shape[0], timesteps - 1, 1)  # Ensure correct shape for LSTM

# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with validation split
history = model_lstm.fit(X, y_seq, epochs=100, batch_size=32, validation_split=0.2)

# Forecasting for the next 52 weeks
future_timestamps_lstm = pd.date_range(start=bangaloreprice.index[-1], periods=52 + 1, freq='W')[1:]
last_sequence_lstm = X[-1]  # Ensure this is correctly shaped as (timesteps-1, 1)

future_predictions_lstm = []
for i in range(52):
    pred_lstm = model_lstm.predict(np.array([last_sequence_lstm]))
    future_predictions_lstm.append(pred_lstm[0, 0])
    last_sequence_lstm = np.roll(last_sequence_lstm, -1)
    last_sequence_lstm[-1] = pred_lstm[0, 0]
    last_sequence_lstm = last_sequence_lstm.reshape(timesteps - 1, 1)  # Ensure correct shape

# Inverse transform the predictions
future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))

# Define your timesteps
timesteps = 50  # Adjust as needed

# Load and prepare the data
y = bangaloreprice.values.flatten()  # Ensure y is a 1D array

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Create sequences
data_timesteps = np.array([y_scaled[i:i + timesteps] for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

# Build RNN model
model_lstm = Sequential()
model_lstm.add(LSTM(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='loss',
                               patience=10,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
                               verbose=1)

# Train the model on the whole dataset
history = model.fit(X, y_seq,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping])

# Prepare the last sequence from the entire data for forecasting
last_sequence = np.array(y_scaled[-(timesteps - 1):]).reshape(1, timesteps - 1, 1)

# Forecast future values for 'future_weeks'
future_weeks = 52
future_timestamps = pd.date_range(start=bangaloreprice.index[-1] + pd.DateOffset(weeks=1),
                                  periods=future_weeks,
                                  freq='W')

future_predictions = []
for _ in range(future_weeks):
    # Predict the next value
    pred = model.predict(last_sequence)[0, 0]
    future_predictions.append(pred)

    # Update the sequence for the next prediction
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = pred

# Convert predictions to a numpy array and reshape for inverse transform
future_predictions = np.array(future_predictions).reshape(-1, 1)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(future_predictions)

# Plotting the forecast for future dates
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index, scaler.inverse_transform(y_scaled), label='Actual Prices', color='blue')
plt.plot(future_timestamps, future_predictions, label='Forecast for Future', color='red')
plt.title('RNN Time Series Forecasting - Future Forecast')
plt.xlabel('Date')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.legend()
plt.show()

"""# DT"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Extract the target variable (y)
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define a range of timesteps to evaluate
timestep_range = range(5, 51)  # Example range from 5 to 50
results = []

for timesteps in timestep_range:
    # Create sequences
    data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps + 1)])

    # Split into features and target
    X, y = data_timesteps[:, :-1], data_timesteps[:, -1]
    train_size = int(len(X) * 0.9)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Flatten the 3D X_train to 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Define the Decision Tree model
    model_dt = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=10)

    # Train the model on the training set
    model_dt.fit(X_train_flat, y_train)

    # Make predictions on the test set
    y_pred_test = model_dt.predict(X_test_flat)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred_test)
    results.append((timesteps, mse))

# Find the optimal timesteps with the lowest MSE
optimal_timesteps, best_mse = min(results, key=lambda x: x[1])

print(f"Optimal Timesteps: {optimal_timesteps}, Best MSE: {best_mse}")

# Re-train the model using the optimal timesteps
timesteps = optimal_timesteps
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps + 1)])
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]
train_size = int(len(X) * 0.9)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Flatten the 3D X_train to 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train the Decision Tree model with optimal timesteps
model_dt = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=10)
model_dt.fit(X_train_flat, y_train)



# Make predictions on the test set
y_pred_test = model_dt.predict(X_test_flat)

# Inverse transform the predictions
y_pred_test = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

# Plot the test predictions
plt.figure(figsize=(15, 6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices (Test)', color='blue')
plt.plot(y_pred_test, label='Predicted Prices (Test)', color='red')
plt.title('Decision Tree Time Series Forecasting - Test Set')
plt.xlabel('Timestamp')
plt.legend()
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Define the model
model_dt = DecisionTreeRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [2, 3, 5, 10, 15, 20, None],  # None means no limit
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=model_dt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model
grid_search.fit(X_train_flat, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")

from sklearn.preprocessing import MinMaxScaler
# Extract the target variable (y)
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 13 # Adjust this value based on your preference

# Create sequences
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps + 1)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

train_size = int(len(X) * 0.9)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Flatten the 3D X_train to 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Limit the depth of the tree
model_dt = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=10)
# Train the model on the training set
model_dt.fit(X_train_flat, y_train)
plt.figure(figsize=(20, 10))
plot_tree(model_dt, filled=True)  # Use feature_names=None since we don't have column names
plt.savefig('decision_tree_diagram.png', dpi=300, bbox_inches='tight')  # Save the tree diagram as a PNG file with dpi=300
plt.show()

# Make predictions on the training set
y_pred_train = model_dt.predict(X_train_flat)

# Inverse transform the predictions
y_pred_train = scaler.inverse_transform(y_pred_train.reshape(-1, 1))

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import plot_tree, DecisionTreeClassifier
import matplotlib.pyplot as plt

# Plot the training predictions
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index[timesteps - 1:train_size + timesteps - 1], scaler.inverse_transform(y)[:train_size], label='Actual Prices', color='blue')
plt.plot(bangaloreprice.index[timesteps - 1:train_size + timesteps - 1], y_pred_train, label='Predicted Prices (Training)', color='red')
plt.title('Decision Tree Time Series Forecasting - Training Set')
plt.xlabel('Timestamp')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred_test = model_dt.predict(X_test_flat)

# Inverse transform the predictions
y_pred_test = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

# Plot the test predictions
plt.figure(figsize=(15, 6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices (Test)', color='blue')
plt.plot(y_pred_test, label='Predicted Prices (Test)', color='red')
plt.title('Decision Tree Time Series Forecasting - Test Set')
plt.xlabel('Timestamp')
plt.legend()
plt.show()

mse = mean_squared_error(scaler.inverse_transform(y_test), y_pred_test)
print('MSE: '+str(mse))
mae = mean_absolute_error(scaler.inverse_transform(y_test), y_pred_test)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test), y_pred_test))
print('RMSE: '+str(rmse))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
print('MAPE for test data: ', mape(scaler.inverse_transform(y_test), y_pred_test))

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
# Absolute metrics
mse = mean_squared_error(scaler.inverse_transform(y_test), y_pred_test)
mae = mean_absolute_error(scaler.inverse_transform(y_test), y_pred_test)
rmse = np.sqrt(mse)

# Relative metrics
mape = mape(y_pred_test,scaler.inverse_transform(y_test))
smape_value = smape(scaler.inverse_transform(y_test), y_pred_test)

# Directional accuracy
correct_directions = np.sign(scaler.inverse_transform(y_test)[1:] - scaler.inverse_transform(y_test)[:-1]) == np.sign(y_pred_test[1:] - y_pred_test[:-1])
directional_accuracy = np.mean(correct_directions)

# Print results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'SMAPE: {smape_value:.2f}%')
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

test = bangaloreprice.index[train_size + timesteps:]

y_test = scaler.inverse_transform(y_test)
y_test_original_flat = y_test.flatten()
y_test_pred_flat = y_pred_test.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat[:-1],
    'Predicted': y_test_pred_flat[:-1]
}, index=test)

predicted

predicted.to_csv('DT_pred.csv')

# Extract the target variable (y)
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define a range of timesteps to evaluate
timestep_range = range(5, 51)  # Example range from 5 to 50
results = []

for timesteps in timestep_range:
    # Create sequences
    data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps)])

    # Split into features and target
    X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

    # Flatten the 3D X to 2D
    X_flat = X.reshape(X.shape[0], -1)

    # Define the Decision Tree model with baseline parameters
    model_dt = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=10)

    # Train the model on the entire dataset
    model_dt.fit(X_flat, y_seq)

    # Make predictions on the same data (this is just to compare timesteps)
    y_pred = model_dt.predict(X_flat)

    # Calculate the mean squared error
    mse = mean_squared_error(y_seq, y_pred)
    results.append((timesteps, mse))

# Find the optimal timesteps with the lowest MSE
optimal_timesteps, best_mse = min(results, key=lambda x: x[1])

print(f"Optimal Timesteps: {optimal_timesteps}, Best MSE: {best_mse}")

# Plot the MSE across different timesteps
timesteps, mses = zip(*results)
plt.plot(timesteps, mses, marker='o')
plt.xlabel('Timesteps')
plt.ylabel('MSE')
plt.title('MSE vs. Timesteps for Decision Tree')
plt.show()

optimal_timesteps

# Extract the target variable (y)
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Create sequences with the optimal number of timesteps
data_timesteps = np.array([[y_scaled[i + j] for j in range(optimal_timesteps)] for i in range(len(y_scaled) - optimal_timesteps + 1)])

# Split into features and target
X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

# Flatten the 3D X to 2D
X_flat = X.reshape(X.shape[0], -1)

# Define the model
model_dt = DecisionTreeRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [2, 3, 5, 10, 15, 20, None],  # None means no limit
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=model_dt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model on the entire dataset with the optimal timesteps
grid_search.fit(X_flat, y_seq)

# Get the best parameters
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")

y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Create sequences with the optimal number of timesteps
data_timesteps = np.array([[y_scaled[i + j] for j in range(optimal_timesteps)] for i in range(len(y_scaled) - optimal_timesteps + 1)])

# Split into features and target
X, y_seq = data_timesteps[:, :-1], data_timesteps[:, -1]

# Flatten the 3D X to 2D
X_flat = X.reshape(X.shape[0], -1)

# Train the model on the full data using the best parameters from grid search
model_dt_full = DecisionTreeRegressor(random_state=42,
                                      max_depth=best_params['max_depth'],
                                      min_samples_split=best_params['min_samples_split'],
                                      min_samples_leaf=best_params['min_samples_leaf'])

model_dt_full.fit(X_flat, y_seq)

# Forecasting for future 52 weeks
future_weeks = 52
future_timestamps = pd.date_range(start=bangaloreprice.index[-1], periods=future_weeks + 1, freq='W')[1:]
last_sequence = np.array(y_scaled[-optimal_timesteps + 1:])

future_predictions = []

for i in range(future_weeks):
    pred = model_dt_full.predict(last_sequence.reshape(1, -1))
    future_predictions.append(pred[0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = pred

# Inverse transform the predictions to get actual values
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Print or plot the future predictions
print(f"Future Predictions: {future_predictions}")

# Combine historical data with forecasted data for plotting
historical_data = bangaloreprice.values.flatten()  # Actual historical data
forecasted_data = np.concatenate([historical_data, future_predictions.flatten()])

# Create a combined index for plotting
combined_index = np.concatenate([bangaloreprice.index, future_timestamps])

# Plot the historical and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(bangaloreprice.index, historical_data, label='Historical Data')
plt.plot(future_timestamps, future_predictions, label='Forecasted Data', linestyle='--')
plt.axvline(x=bangaloreprice.index[-1], color='r', linestyle=':', label='Forecast Start')
plt.title('Historical and Forecasted Apple Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

"""# RF"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Extract the target variable (y)
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define a range of timesteps to evaluate
timestep_range = range(5, 51)  # Example range from 10 to 50
results = []

for timesteps in timestep_range:
    # Create sequences
    data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps + 1)])

    # Split into features and target
    X, y = data_timesteps[:, :-1], data_timesteps[:, -1]
    train_size = int(len(X) * 0.9)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Flatten the 2D X_train to 1D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Define the Random Forest model
    model_rf = RandomForestRegressor(random_state=42, n_estimators=50,min_samples_leaf = 1, min_samples_split = 10, max_depth=10)

    # Train the model on the training set
    model_rf.fit(X_train_flat, y_train)

    # Make predictions on the test set
    y_pred_test = model_rf.predict(X_test_flat)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred_test)
    results.append((timesteps, mse))

# Find the optimal timesteps with the lowest MSE
optimal_timesteps, best_mse = min(results, key=lambda x: x[1])

print(f"Optimal Timesteps: {optimal_timesteps}, Best MSE: {best_mse}")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Extract the target variable (y)
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 8

# Create sequences
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps + 1)])
X_past, y_past = data_timesteps[:, :-1], data_timesteps[:, -1]

# Split the data into training and testing sets
train_size = int(len(X_past) * 0.9)
X_past_train, X_past_test, y_past_train, y_past_test = X_past[:train_size], X_past[train_size:], y_past[:train_size], y_past[train_size:]

# Flatten the data for the model
X_past_train_flat = X_past_train.reshape(X_past_train.shape[0], -1)
X_past_test_flat = X_past_test.reshape(X_past_test.shape[0], -1)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a RandomForestRegressor
model_rf_past = RandomForestRegressor(random_state=42)

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(model_rf_past, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_past_train_flat, y_past_train)

# Get the best parameters and estimator
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
model_rf_past = grid_search.best_estimator_

# Train the model on the training set
model_rf_past.fit(X_past_train_flat, y_past_train)

# Predictions on the training set
y_past_train_pred = model_rf_past.predict(X_past_train_flat)

# Predictions on the test set
y_past_test_pred = model_rf_past.predict(X_past_test_flat)

# Inverse transform the predictions
y_past_train_pred = scaler.inverse_transform(y_past_train_pred.reshape(-1, 1))
y_past_test_pred = scaler.inverse_transform(y_past_test_pred.reshape(-1, 1))

# Ensure that y_past_train and y_past_test are also scaled correctly before inverse transformation
y_past_train = scaler.inverse_transform(y_past_train.reshape(-1, 1))
y_past_test = scaler.inverse_transform(y_past_test.reshape(-1, 1))

# Output predictions
print("Training Predictions:", y_past_train_pred)
print("Testing Predictions:", y_past_test_pred)

# Plot the training predictions vs actual training data
plt.figure(figsize=(15, 6))
plt.plot(y_past_train, label='Actual Prices (Train)', color='blue')
plt.plot(y_past_train_pred, label='Predicted Prices (Train)', color='red')
plt.title('Random Forest Time Series Forecasting - Training Set')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_past_test, label='Actual Prices (Test)', color='blue')
plt.plot(y_past_test_pred, label='Predicted Prices (Test)', color='red')
plt.title('Random Forest Time Series Forecasting - Test Set')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.legend()
plt.show()

mse = mean_squared_error(y_past_test, y_past_test_pred)
print('MSE: '+str(mse))
mae = mean_absolute_error(y_past_test, y_past_test_pred)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(y_past_test, y_past_test_pred))
print('RMSE: '+str(rmse))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
print('MAPE for test data: ', mape(y_past_test, y_past_test_pred))

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
# Absolute metrics
mse = mean_squared_error(y_past_test, y_past_test_pred)
mae = mean_absolute_error(y_past_test, y_past_test_pred)
rmse = np.sqrt(mse)

# Relative metrics
mape = mape(y_past_test_pred,y_past_test)
smape_value = smape(y_past_test, y_past_test_pred)

# Directional accuracy
correct_directions = np.sign(y_past_test[1:] - y_past_test[:-1]) == np.sign(y_past_test_pred[1:] - y_past_test_pred[:-1])
directional_accuracy = np.mean(correct_directions)

# Print results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'SMAPE: {smape_value:.2f}%')
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

test_dates = bangaloreprice.index[-len(y_past_test):]

# Create a DataFrame with the actual and predicted prices
df_test_predictions = pd.DataFrame({
    'Date': test_dates,
    'Actual Price': y_past_test.flatten(),
    'Predicted Price': y_past_test_pred.flatten()
})

# Set the 'Date' column as the index
df_test_predictions.set_index('Date', inplace=True)

# Display the resulting DataFrame
print(df_test_predictions)

df_test_predictions.to_csv('RF_pred.csv')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Extract the target variable (y) from mechuaprice
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 7

# Use the trained RandomForest model to forecast for the future
future_weeks = 52
future_timestamps = pd.date_range(start=bangaloreprice.index[-1], periods=future_weeks + 1, freq='W')[1:]

# Start with the last sequence of 5 timesteps from the existing data
last_sequence = np.array(y_scaled[-timesteps:]).reshape(1, -1)  # Use the last 5 timesteps

future_predictions = []

for i in range(future_weeks):
    # Predict the next value
    pred = model_rf_past.predict(last_sequence)
    future_predictions.append(pred[0])

    # Update the last_sequence to include the new prediction
    last_sequence = np.append(last_sequence[:, 1:], pred).reshape(1, -1)  # Shift left and add prediction

# Inverse transform the predictions to get them back to the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plotting the actual prices and the forecast for the future
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index, y, label='Actual Prices', color='blue')
plt.plot(future_timestamps, future_predictions, label='Forecast for Future', color='red')
plt.title('Random Forest Time Series Forecasting - Future Forecast')
plt.xlabel('Date')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
# Assuming 'parimporeprice' is your DataFrame containing the price column
y = bangaloreprice.values
# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))
# Define the number of timesteps
timesteps = 10
# Create sequences
data_timesteps = np.array([[y_scaled[i+j] for j in range(timesteps)] for i in range(len(y_scaled)-timesteps+1)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Flatten X
X = X.reshape(X.shape[0], -1)

# Split data into training and test sets (last 10%)
split_index = int(0.9 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Extract dates for the test set
dates = bangaloreprice.index
test_dates = dates[timesteps:][split_index+timesteps:]
# Define and fit the Bagging Regressor with SVR as the base estimator
bagging_regressor = BaggingRegressor(base_estimator=SVR(kernel='linear', gamma='scale', C=1, epsilon=0.01),
                                     n_estimators=10, random_state=42)
bagging_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = bagging_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE of Bagging Regressor: {mse}')

# Grid Search for optimal parameters
param_grid = {
    'base_estimator__C': [10, 50, 100],
    'base_estimator__epsilon': [0.01, 0.05, 0.1],
    'n_estimators': [10, 20, 50]
}

grid_search = GridSearchCV(estimator=bagging_regressor, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {-grid_search.best_score_}')

# Extract the best parameters from grid search
best_params = grid_search.best_params_

# Create a new Bagging Regressor with the best parameters
best_bagging_regressor = BaggingRegressor(
    base_estimator=SVR(kernel='linear', gamma='scale', C=best_params['base_estimator__C'], epsilon=best_params['base_estimator__epsilon']),
    n_estimators=best_params['n_estimators'],
    random_state=42
)

# Fit the model with the best parameters
best_bagging_regressor.fit(X_train, y_train)

# Predict and evaluate with the refitted model
y_pred_best = best_bagging_regressor.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print(f'MSE of Bagging Regressor with Best Parameters: {mse_best}')

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

# Define a pipeline for scaling and model
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('bagging', BaggingRegressor(base_estimator=SVR(), random_state=42))
])
# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'bagging__base_estimator__C': [10, 50, 100],
    'bagging__base_estimator__epsilon': [0.01, 0.05, 0.1],
    'bagging__n_estimators': [10, 20, 50, 100],
    'bagging__max_samples': [0.5, 0.7, 1.0],
    'bagging__max_features': [0.5, 0.7, 1.0]
}

# Perform Randomized Search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_distributions,
                                    n_iter=50, cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
                                    random_state=42)
random_search.fit(X_train, y_train)

# Extract the best parameters and refit the model
best_pipeline = random_search.best_estimator_
y_pred_best = best_pipeline.predict(X_test)

# Inverse transform and calculate metrics
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred_best.reshape(-1, 1))

mse_best = mean_squared_error(y_test_original, y_pred_original)
mae_best = mean_absolute_error(y_test_original, y_pred_original)
rmse_best = np.sqrt(mse_best)
mape_best = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

print(f'MSE: {mse_best}')
print(f'MAE: {mae_best}')
print(f'RMSE: {rmse_best}')
print(f'MAPE: {mape_best}%')

plt.figure(figsize=(10, 3))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices', color='blue', marker='o')
plt.plot(scaler.inverse_transform(y_pred_best.reshape(-1, 1)), label='Predicted Prices', color='red', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices (Test Set)')
plt.legend()
plt.show()

# Extract the best parameters and refit the model
best_pipeline = random_search.best_estimator_
y_pred_scaled = best_pipeline.predict(X_test)

# Calculate performance metrics in the scaled format
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)
mape_scaled = np.mean(np.abs((y_test - y_pred_scaled) / y_test)) * 100

print(f'MSE (Scaled): {mse_scaled}')
print(f'MAE (Scaled): {mae_scaled}')
print(f'RMSE (Scaled): {rmse_scaled}')
print(f'MAPE (Scaled): {mape_scaled}%')

dates = bangaloreprice.index
test_dates = dates[timesteps:][split_index+timesteps:]

test_dates = dates[-25:]

y_test_original_flat = y_test_original.flatten()
y_test_pred_flat = y_pred_original.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
}, index= test_dates)

predicted

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define a function to create and train an LSTM model
def create_and_train_lstm(X_train, y_train, timesteps):
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', input_shape=(timesteps-1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model

# Assuming 'parimporeprice' is your DataFrame containing the price column
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 10
# Create sequences
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Reshape X for LSTM (which expects 3D data)
X = X.reshape(X.shape[0], timesteps - 1, 1)
train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create multiple LSTM models (bagging)
num_models = 15
models = []
for _ in range(num_models):
    # Sample a subset of the training data
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_train_subset, y_train_subset = X_train[indices], y_train[indices]

    # Create and train LSTM model
    model = create_and_train_lstm(X_train_subset, y_train_subset, timesteps)
    models.append(model)

# Predict with each model and average the predictions
predictions = np.zeros((len(X_test), num_models))
for i, model in enumerate(models):
    predictions[:, i] = model.predict(X_test).flatten()

y_pred_lstm = np.mean(predictions, axis=1)

# Inverse transform for original scale
y_pred_lstm_original = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

# Calculate metrics
mse_lstm = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_lstm_original)
mae_lstm = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_lstm_original)
rmse_lstm = np.sqrt(mse_lstm)
mape_lstm = np.mean(np.abs((scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() - y_pred_lstm_original) / scaler.inverse_transform(y_test.reshape(-1, 1)).flatten())) * 100

print(f'MSE (LSTM Bagging): {mse_lstm}')
print(f'MAE (LSTM Bagging): {mae_lstm}')
print(f'RMSE (LSTM Bagging): {rmse_lstm}')
print(f'MAPE (LSTM Bagging): {mape_lstm}%')

# Plot actual vs. predicted values for LSTM Bagging
plt.figure(figsize=(10, 3))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices', color='blue', marker='o')
plt.plot(y_pred_lstm_original, label='Predicted Prices', color='red', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices (LSTM Bagging)')
plt.legend()
plt.show()

y_test = scaler.inverse_transform(y_test)
y_test_original_flat = y_test.flatten()
y_test_pred_flat = y_pred_lstm_original.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
}, index=test_data.index)

predicted

# Forecasting for next 52 weeks
future_timestamps_lstm = pd.date_range(start=bangaloreprice.index[-1], periods=52 + 1, freq='W')[1:]
last_sequence_lstm = X[-1]

future_predictions_lstm = []
for i in range(52):
    pred_lstm = model.predict(np.array([last_sequence_lstm]))
    future_predictions_lstm.append(pred_lstm[0, 0])
    last_sequence_lstm = np.roll(last_sequence_lstm, -1)
    last_sequence_lstm[-1] = pred_lstm[0, 0]

# Inverse transform the predictions
future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))

# Plotting the forecast for future dates using LSTM
plt.figure(figsize=(15, 6))
plt.plot(bangaloreprice.index, bangaloreprice.values, label='Original Data', color='blue')
plt.plot(future_timestamps_lstm, future_predictions_lstm, label='LSTM Forecast for Future', color='red')
plt.legend()
plt.title('Forecasting for Next 52 Weeks using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor

# Assuming 'parimporeprice' is your DataFrame containing the price column
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 10
# Create sequences
data_timesteps = np.array([[y_scaled[i + j] for j in range(timesteps)] for i in range(len(y_scaled) - timesteps)])

# Split into features and target
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Flatten X for Decision Tree
X = X.reshape(X.shape[0], -1)  # Flattening the data for Decision Tree

# Split data into training and test sets (last 25%)
train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train Decision Tree Regressor with Bagging
bagging_dt = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)
bagging_dt.fit(X_train, y_train)
y_pred_dt = bagging_dt.predict(X_test)

# Hyperparameter tuning for Decision Tree Regressor
param_grid = {
    'base_estimator__max_depth': [3, 5, 10],
    'base_estimator__min_samples_split': [2, 5, 10],
    'base_estimator__min_samples_leaf': [1, 2, 4],
    'n_estimators': [5 , 10, 20]
}

grid_search = GridSearchCV(BaggingRegressor(base_estimator=DecisionTreeRegressor(), random_state=42),
                           param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters for Decision Tree Regressor:", best_params)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV

# Create a base Decision Tree model with constraints
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=2)

# Create the Bagging Regressor with the constrained Decision Tree
bagging_dt = BaggingRegressor(base_estimator=dt, n_estimators=5, max_features=0.8, random_state=42)

# Fit the model
bagging_dt.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = bagging_dt.predict(X_test)
y_pred_best_dt_original = scaler.inverse_transform(y_pred_dt.reshape(-1, 1)).flatten()
# Calculate metrics
mse_dt = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_best_dt_original)
mae_dt = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_best_dt_original)
rmse_dt = np.sqrt(mse_dt)
mape_dt = np.mean(np.abs((scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() - y_pred_best_dt_original) / scaler.inverse_transform(y_test.reshape(-1, 1)).flatten())) * 100

print(f'MSE (Decision Tree Bagging): {mse_dt}')
print(f'MAE (Decision Tree Bagging): {mae_dt}')
print(f'RMSE (Decision Tree Bagging): {rmse_dt}')
print(f'MAPE (Decision Tree Bagging): {mape_dt}%')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 3))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices', color='blue', marker='o')
plt.plot(y_pred_best_dt_original, label='Predicted Prices (DT Bagging)', color='red', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices (Decision Tree Bagging)')
plt.legend()
plt.show()

y_test_original_flat = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred_flat = y_pred_best_dt_original.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
})

predicted

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to create sequences for LSTM
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

# Scale data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))  # Assuming y is your target variable

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Assuming 'parimporeprice' is your DataFrame containing the price column
y = bangaloreprice.values

# Scale the data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Define the number of timesteps
timesteps = 10

# Create sequences for time series data
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

# Create sequences for the scaled data
data_timesteps = create_sequences(y_scaled, timesteps)
X, y = data_timesteps[:, :-1], data_timesteps[:, -1]

# Split into training and testing sets (90% training, 10% testing)
train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X for LSTM (which expects 3D data)
X_train_lstm = X_train.reshape(X_train.shape[0], timesteps - 1, 1)
X_test_lstm = X_test.reshape(X_test.shape[0], timesteps - 1, 1)

# Ensure that y_train and y_test are 1D arrays
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Flatten X_train and X_test for SVR and Decision Tree
X_train_svr_dt = X_train.reshape(X_train.shape[0], -1)  # (196, 39)
X_test_svr_dt = X_test.reshape(X_test.shape[0], -1)    # Adjust shape as needed

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.tree import DecisionTreeRegressor

def custom_boosting(models, X_train, X_test, X_test_lstm, y_train, y_test, scaler, timesteps):
    predictions = np.zeros((len(X_test), len(models)))

    for i, model in enumerate(models):
        if isinstance(model, ARIMA):
            # ARIMA prediction requires specific handling
            try:
                # Predict with ARIMA
                forecast = model.get_forecast(steps=len(y_test))
                predictions[:, i] = forecast.predicted_mean
            except Exception as e:
                print(f"ARIMA Prediction Error: {e}")
                predictions[:, i] = np.nan  # Handle errors gracefully
        elif isinstance(model, Sequential):  # LSTM model
            try:
                # Predict with LSTM
                predictions[:, i] = model.predict(X_test_lstm).flatten()
            except Exception as e:
                print(f"LSTM Prediction Error: {e}")
                predictions[:, i] = np.nan  # Handle errors gracefully
        else:  # SVR, Decision Tree
            try:
                # Adjust test input shape for SVR and Decision Tree
                X_test_model = X_test.reshape(X_test.shape[0], -1) if not isinstance(model, LSTM) else X_test_lstm
                predictions[:, i] = model.predict(X_test_model).flatten()
            except Exception as e:
                print(f"SVR/Decision Tree Prediction Error: {e}")
                predictions[:, i] = np.nan  # Handle errors gracefully

    # Combine predictions (simple average)
    combined_predictions = np.nanmean(predictions, axis=1)  # Use mean ignoring NaNs

    # Rescale predictions back to original scale
    combined_predictions_original = scaler.inverse_transform(combined_predictions.reshape(-1, 1)).flatten()

    # Calculate error metrics
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test_original, combined_predictions_original)
    mae = mean_absolute_error(y_test_original, combined_predictions_original)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_original - combined_predictions_original) / y_test_original)) * 100

    return combined_predictions_original, mse, mae, rmse, mape

# Fit the ARIMA model separately
arima_model = ARIMA(y_scaled[:train_size], order=(2, 1, 1))
arima_model_fitted = arima_model.fit()

# Fit the SVR model
svr_model = SVR(kernel='rbf', gamma='scale', C=0.3, epsilon=0.01)
X_train_svr_dt = X_train.reshape(X_train.shape[0], -1)  # Flattened for SVR
X_test_svr_dt = X_test.reshape(X_test.shape[0], -1)    # Flattened for SVR
svr_model.fit(X_train_svr_dt, y_train)

# Fit the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, activation='relu', input_shape=(timesteps - 1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)

# Fit the Decision Tree model
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=10)
dt_model.fit(X_train_svr_dt, y_train)

# Prepare test data for LSTM
X_test_lstm = X_test.reshape(X_test.shape[0], timesteps - 1, 1)  # Adjust based on LSTM input

# Combine models
models = [arima_model_fitted, svr_model, lstm_model, dt_model]

# Get predictions and evaluate
boosted_predictions, boosted_mse, boosted_mae, boosted_rmse, boosted_mape = custom_boosting(
    models, X_train, X_test, X_test_lstm, y_train, y_test, scaler, timesteps
)

# Print the performance metrics
print(f'Boosted Model MSE: {boosted_mse}')
print(f'Boosted Model MAE: {boosted_mae}')
print(f'Boosted Model RMSE: {boosted_rmse}')
print(f'Boosted Model MAPE: {boosted_mape}')

# Plot the results
plt.figure(figsize=(10, 3))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices', color='blue')
plt.plot(boosted_predictions, label='Boosted Predictions', color='red', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Actual vs. Boosted Predictions')
plt.legend()
plt.show()

y_test_original_flat = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred_flat = boosted_predictions.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
})

predicted

import numpy as np

def calculate_directional_accuracy(actual_values, predicted_values):
    """
    Calculate the directional accuracy of predictions.

    Parameters:
    actual_values (array-like): Actual values of the time series.
    predicted_values (array-like): Forecasted values of the time series.

    Returns:
    float: Directional accuracy as a percentage.
    """
    # Ensure inputs are numpy arrays
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    # Calculate changes
    actual_changes = np.diff(actual_values)
    predicted_changes = np.diff(predicted_values)

    # Determine direction (1 for increase, -1 for decrease)
    actual_direction = np.sign(actual_changes)
    predicted_direction = np.sign(predicted_changes)

    # Count correct predictions
    correct_direction = np.sum(actual_direction == predicted_direction)

    # Calculate directional accuracy
    total_comparisons = len(actual_direction)
    directional_accuracy = correct_direction / total_comparisons

    return directional_accuracy * 100  # Convert to percentage

# Assuming y_test and boosted_predictions are your actual and predicted values
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # Inverse transform the actual values
boosted_predictions_original = scaler.inverse_transform(boosted_predictions.reshape(-1, 1)).flatten()  # Inverse transform the predictions

# Calculate directional accuracy
accuracy = calculate_directional_accuracy(y_test_original, boosted_predictions_original)
print(f"Directional Accuracy: {accuracy:.2f}%")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Fit the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_svr_dt, y_train)  # X_train should be reshaped for non-LSTM models

# Make predictions on the test set
gb_predictions = gb_model.predict(X_test_svr_dt)

# Rescale predictions back to the original scale
gb_predictions_original = scaler.inverse_transform(gb_predictions.reshape(-1, 1)).flatten()
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate performance metrics
gb_mse = mean_squared_error(y_test_original, gb_predictions_original)
gb_mae = mean_absolute_error(y_test_original, gb_predictions_original)
gb_rmse = np.sqrt(gb_mse)
gb_mape = np.mean(np.abs((y_test_original - gb_predictions_original) / y_test_original)) * 100

# Print the performance metrics
print(f'Gradient Boosting Model MSE: {gb_mse}')
print(f'Gradient Boosting Model MAE: {gb_mae}')
print(f'Gradient Boosting Model RMSE: {gb_rmse}')
print(f'Gradient Boosting Model MAPE: {gb_mape}%')

# Plot the results
plt.figure(figsize=(10, 3))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices', color='blue')
plt.plot(gb_predictions_original, label='Gradient Boosting Predictions', color='red', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Actual vs. Gradient Boosting Predictions')
plt.legend()
plt.show()



y_test_original_flat = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred_flat = gb_predictions_original.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
})

predicted

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

# Function to train and predict with base models
def fit_predict_base_models(X_train, y_train, X_test):
    # Train ARIMA
    arima_model = ARIMA(y_train, order=(2, 1, 1))
    arima_model_fitted = arima_model.fit()
    arima_preds = arima_model_fitted.get_forecast(steps=len(X_test)).predicted_mean

    # Train SVR
    svr_model = SVR(kernel='linear', gamma='scale', C=0.03, epsilon=0.01)
    svr_model.fit(X_train, y_train)
    svr_preds = svr_model.predict(X_test)

    # Train LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=50, batch_size=16, verbose=0)
    lstm_preds = lstm_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()

    # Train Decision Tree
    dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10,min_samples_leaf=2, min_samples_split=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    return np.column_stack((arima_preds, svr_preds, lstm_preds, dt_preds, rf_preds))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

y = bangaloreprice.values

# Prepare the data
timesteps = 26  # Example value; adjust based on your use case
X, y = [], []

# Generate sequences of data
for i in range(len(bangaloreprice) - timesteps):
    X.append(bangaloreprice.values[i:i + timesteps])
    y.append(bangaloreprice.values[i + timesteps])

X = np.array(X)
y = np.array(y)

# Split the data
train_size = int(len(y) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Fit the ARIMA model
arima_model = ARIMA(y[:X_train.shape[0]], order=(2, 1, 1))
arima_model_fitted = arima_model.fit()

# Fit the SVR model
svr_model = SVR(kernel='linear', gamma='scale', C=0.03, epsilon=0.01)
svr_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Fit the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, activation='relu', input_shape=(timesteps, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)

# Fit the Decision Tree model
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5)
dt_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Make predictions
arima_predictions = arima_model_fitted.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
svr_predictions = svr_model.predict(X_test.reshape(X_test.shape[0], -1))
lstm_predictions = lstm_model.predict(X_test_lstm).flatten()
dt_predictions = dt_model.predict(X_test.reshape(X_test.shape[0], -1))

# Stack the predictions
base_model_preds = np.column_stack([svr_predictions, lstm_predictions, dt_predictions])

# Use Random Forest as the meta-learner
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
meta_model.fit(base_model_preds, y_test)
final_predictions = meta_model.predict(base_model_preds)

# Evaluate performance
mse = mean_squared_error(y_test, final_predictions)
mae = mean_absolute_error(y_test, final_predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - final_predictions) / y_test)) * 100

print(f'Stacking Model MSE: {mse}')
print(f'Stacking Model MAE: {mae}')
print(f'Stacking Model RMSE: {rmse}')
print(f'Stacking Model MAPE: {mape}')

# Plot results
plt.figure(figsize=(10, 3))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(final_predictions, label='Stacking Model Predictions', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Actual vs Stacking Model Predictions')
plt.legend()
plt.show()

y_test_original_flat = y_test.flatten()
y_test_pred_flat = final_predictions.flatten()
predicted = pd.DataFrame({
    'Actual': y_test_original_flat,
    'Predicted': y_test_pred_flat
})

predicted

# Number of future data points to forecast (25 in this case)
future_steps = 8

# Create an empty list to store future predictions
future_predictions_svr = []
future_predictions_lstm = []
future_predictions_dt = []
future_predictions_arima = []

# Start with the last available sequence from the test set or training data for prediction
last_sequence = X_test[-1]  # or X_train[-1], based on your strategy

# Reshape the last sequence to match LSTM input shape
last_sequence_lstm = last_sequence.reshape(1, timesteps, 1)

for step in range(future_steps):
    # Forecasting with SVR
    next_pred_svr = svr_model.predict(last_sequence.reshape(1, -1))
    future_predictions_svr.append(next_pred_svr[0])

    # Forecasting with LSTM
    next_pred_lstm = lstm_model.predict(last_sequence_lstm)
    future_predictions_lstm.append(next_pred_lstm[0, 0])

    # Forecasting with Decision Tree
    next_pred_dt = dt_model.predict(last_sequence.reshape(1, -1))
    future_predictions_dt.append(next_pred_dt[0])

    # Forecasting with ARIMA (using previous predictions as inputs)
    next_pred_arima = arima_model_fitted.predict(start=len(y_train) + len(y_test) + step, end=len(y_train) + len(y_test) + step)
    future_predictions_arima.append(next_pred_arima[0])

    # Update the last sequence with the newly predicted value (sliding window approach)
    last_sequence = np.roll(last_sequence, -1)  # Shift the sequence to the left
    last_sequence[-1] = next_pred_svr  # Add the predicted value at the end

    # Update the LSTM sequence in the same way
    last_sequence_lstm = last_sequence.reshape(1, timesteps, 1)

# Stack future predictions from SVR, LSTM, Decision Tree
future_base_preds = np.column_stack([future_predictions_svr, future_predictions_lstm, future_predictions_dt])

# Meta model to predict the final ensemble forecast for future data points
future_final_predictions = meta_model.predict(future_base_preds)

# Output future predictions
future_final_predictions = np.array(future_final_predictions)

# Print the future forecast for the months in 2023
print(f'Future 25 Predictions for 2023 using Stacked Model: {future_final_predictions}')

# Plot the future forecasted prices for 2023
plt.figure(figsize=(12, 6))
plt.plot(future_final_predictions, label='Forecasted Prices for 2023', color='green', linestyle='--')
plt.xlabel('Future Time Steps')
plt.ylabel('Price')
plt.title('Forecasted Prices for 2023 (Months 1, 2, 9, 10, 11, 12)')
plt.legend()
plt.show()

# Ensure y is a 1D array by flattening if necessary
y_flat = y.flatten()

# Concatenate actual data and future predictions
extended_data = np.concatenate([y_flat, future_final_predictions])

# Create a time index for the actual data and future forecast
time_index_actual = np.arange(1, len(y_flat) + 1)  # Time steps for actual data
time_index_future = np.arange(len(y_flat) + 1, len(y_flat) + future_steps + 1)  # Time steps for future forecast

# Plot the actual data along with the future forecasted data
plt.figure(figsize=(14, 7))

# Plot actual data
plt.plot(time_index_actual, y_flat, label='Actual Prices', color='blue')

# Plot future forecasted data
plt.plot(time_index_future, future_final_predictions, label='Forecasted Prices for 2023', color='red', linestyle='--')


# Set plot labels and title
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()

# Show the plot
plt.show()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
np.random.seed(32)
random.seed(32)
tf.random.set_seed(32)
# Assuming bangaloreprice is your dataset
y = bangaloreprice.values

# Prepare the data
timesteps = 40  # Example value; adjust based on your use case
X, y_seq = [], []

# Generate sequences of data
for i in range(len(bangaloreprice) - timesteps):
    X.append(bangaloreprice.values[i:i + timesteps])
    y_seq.append(bangaloreprice.values[i + timesteps])

X = np.array(X)
y_seq = np.array(y_seq)

# Reshape data for LSTM
X_lstm = X.reshape(X.shape[0], X.shape[1], 1)

# Fit the ARIMA model on the entire data
arima_model = ARIMA(y_seq, order=(2, 1, 1))
arima_model_fitted = arima_model.fit()

# Fit the SVR model
svr_model = SVR(kernel='rbf', gamma='scale', C=10, epsilon=0.01)
svr_model.fit(X.reshape(X.shape[0], -1), y_seq)

# Fit the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, activation='relu', input_shape=(timesteps, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_lstm, y_seq, epochs=100, batch_size=32, verbose=0)

# Fit the Decision Tree model
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5)
dt_model.fit(X.reshape(X.shape[0], -1), y_seq)

# Now forecast the next 8 weeks (future_steps)
future_steps = 8

# Create an empty list to store future predictions
future_predictions_svr = []
future_predictions_lstm = []
future_predictions_dt = []
future_predictions_arima = []

# Start with the last available sequence from the data for prediction
last_sequence = X[-1]  # Last sequence of data
last_sequence_lstm = last_sequence.reshape(1, timesteps, 1)

for step in range(future_steps):
    # Forecasting with SVR
    next_pred_svr = svr_model.predict(last_sequence.reshape(1, -1))
    future_predictions_svr.append(next_pred_svr[0])

    # Forecasting with LSTM
    next_pred_lstm = lstm_model.predict(last_sequence_lstm)
    future_predictions_lstm.append(next_pred_lstm[0, 0])

    # Forecasting with Decision Tree
    next_pred_dt = dt_model.predict(last_sequence.reshape(1, -1))
    future_predictions_dt.append(next_pred_dt[0])

    # Forecasting with ARIMA (using previous predictions as inputs)
    next_pred_arima = arima_model_fitted.predict(start=len(y_seq) + step, end=len(y_seq) + step)
    future_predictions_arima.append(next_pred_arima[0])

    # Update the last sequence with the newly predicted value (sliding window approach)
    last_sequence = np.roll(last_sequence, -1)  # Shift the sequence to the left
    last_sequence[-1] = next_pred_svr  # Add the predicted value at the end
    last_sequence_lstm = last_sequence.reshape(1, timesteps, 1)

# Stack future predictions from SVR, LSTM, Decision Tree
future_base_preds = np.column_stack([future_predictions_svr, future_predictions_dt, future_predictions_arima])

# Meta model to predict the final ensemble forecast for future data points
meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
meta_model.fit(future_base_preds, future_predictions_lstm)  # Meta learner trained on the base predictions
future_final_predictions = meta_model.predict(future_base_preds)

# Output future predictions
print(f'Future 8 Week Predictions: {future_final_predictions}')

# Plot the future forecasted prices for the next 8 weeks
plt.figure(figsize=(12, 6))
plt.plot(future_final_predictions, label='Forecasted Prices (Next 8 Weeks)', color='green', linestyle='--')
plt.xlabel('Future Time Steps')
plt.ylabel('Price')
plt.title('Forecasted Prices for Next 8 Weeks')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

# Assuming 'parimporeprice' is your input DataFrame containing weekly data
y = bangaloreprice.values

# Function to create sequences based on timesteps (for SVR, LSTM, Decision Tree)
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps])
    return np.array(X), np.array(y)

# Optimal timesteps for SVR, LSTM, and Decision Tree
timesteps_dict = {'svr': 30, 'lstm': 10, 'dt': 13}

# Prepare data for each model based on the optimal timesteps
X_svr, y_svr = create_sequences(y, timesteps_dict['svr'])
X_lstm, y_lstm = create_sequences(y, timesteps_dict['lstm'])
X_dt, y_dt = create_sequences(y, timesteps_dict['dt'])

# Reshape data for LSTM
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# Fit each base model
# SVR
svr_model = SVR(kernel='rbf', C=0.09, epsilon=0.01, gamma='scale')  # Updated parameters
svr_model.fit(X_svr.reshape(X_svr.shape[0], -1), y_svr)

# LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, activation='relu', input_shape=(timesteps_dict['lstm'], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_lstm, y_lstm, epochs=100, batch_size=32, verbose=0)

# Decision Tree
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=5, min_samples_leaf=5)  # Updated parameters
dt_model.fit(X_dt.reshape(X_dt.shape[0], -1), y_dt)

# ARIMA with order (1, 0, 2)
arima_model = ARIMA(y, order=(2, 1, 2))
arima_model_fitted = arima_model.fit()

# Forecasting for the next 12 weeks
future_steps = 8

# Initialize lists to store future predictions
future_predictions_svr = []
future_predictions_lstm = []
future_predictions_dt = []
future_predictions_arima = []
meta_predictions = []

# Start with the last available sequence for each model
last_sequence_svr = X_svr[-1]
last_sequence_lstm = X_lstm[-1].reshape(1, timesteps_dict['lstm'], 1)
last_sequence_dt = X_dt[-1]

for step in range(future_steps):
    # SVR
    next_pred_svr = svr_model.predict(last_sequence_svr.reshape(1, -1))
    future_predictions_svr.append(next_pred_svr[0])

    # LSTM
    next_pred_lstm = lstm_model.predict(last_sequence_lstm)
    future_predictions_lstm.append(next_pred_lstm[0, 0])

    # Decision Tree
    next_pred_dt = dt_model.predict(last_sequence_dt.reshape(1, -1))
    future_predictions_dt.append(next_pred_dt[0])

    # ARIMA (using model's forecast for future steps)
    next_pred_arima = arima_model_fitted.forecast(steps=1)
    future_predictions_arima.append(next_pred_arima[0])

    # Update sequences with the predicted values
    last_sequence_svr = np.roll(last_sequence_svr, -1)
    last_sequence_svr[-1] = next_pred_svr

    last_sequence_lstm = np.roll(last_sequence_lstm, -1)
    last_sequence_lstm[-1] = next_pred_lstm

    last_sequence_dt = np.roll(last_sequence_dt, -1)
    last_sequence_dt[-1] = next_pred_dt

# Combine the predictions from all models for meta-learning
meta_features = np.array([
    future_predictions_svr,
    future_predictions_lstm,
    future_predictions_dt,
    future_predictions_arima
]).T  # Transpose to get correct shape for meta-learner

# Train a meta-learner (e.g., RandomForestRegressor)
meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
meta_learner.fit(meta_features, np.array(y[-future_steps:]))  # Use actual values for training

# Get meta predictions
meta_predictions = meta_learner.predict(meta_features)

# Print and plot the predictions
print(f'SVR Predictions: {future_predictions_svr}')
print(f'LSTM Predictions: {future_predictions_lstm}')
print(f'Decision Tree Predictions: {future_predictions_dt}')
print(f'ARIMA Predictions: {future_predictions_arima}')
print(f'Meta Learner Predictions: {meta_predictions}')

# Plotting the future forecast
plt.figure(figsize=(12, 6))
plt.plot(future_predictions_svr, label='SVR Forecast', linestyle='--')
plt.plot(future_predictions_lstm, label='LSTM Forecast', linestyle='--')
plt.plot(future_predictions_dt, label='Decision Tree Forecast', linestyle='--')
plt.plot(future_predictions_arima, label='ARIMA Forecast', linestyle='--')
plt.plot(meta_predictions, label='Meta Learner Forecast', linestyle='-', color='black')
plt.xlabel('Future Weeks')
plt.ylabel('Price')
plt.title('Future 12 Weeks Forecast')
plt.legend()
plt.show()
