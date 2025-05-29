# Imports
from tensorflow import keras 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

data = pd.read_csv('MicrosoftStock.csv')

print(data.head())
print(data.info())
print(data.describe())


# Initial Data Visualization
# Plot 1 - Open and Close Prices of time
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['open'], label='Open', color='blue')
plt.plot(data['date'], data['close'], label='Close', color='red')
plt.title("Open and Close Prices Over Time")
plt.legend()
#plt.show()

# Plot 2 - Trading Volume ( check for outliers and trends)
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['volume'], label='Volume', color='orange')
plt.title("Stock Volume Over Time")
#plt.show()

# Drop non-numeric columns
numeric_data = data.select_dtypes(include=("int64", "float64"))

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8, 6)) 

sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
#plt.show()

# Conver the Data into Date time then create a date filter
data['date'] = pd.to_datetime(data['date'])

prediction = data.loc[
    (data['date'] > datetime(2013, 1, 1)) &
    (data['date'] < datetime(2018,1,1))
]

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['close'], color='blue')
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price Over Time")


# Prepare for the LSTM Model (sequential)
stock_close = data.filter(['close'])  # we will only use the close price for prediction

dataset = stock_close.values # convert to numpy array

training_data_len = int(np.ceil(len(dataset) * 0.95))  # 95% for training

# Preprocessing stages 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]  # 95% of all out data 


X_train, y_train = [], []  # the data we will use to train the model


# create a sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])  # previous 60 days
    y_train.append(training_data[i, 0])      # current day


X_train, y_train = np.array(X_train), np.array(y_train)  # were doing this for tensorflow because it handles arrays

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # reshape for LSTM input

# Build the LSTM Model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer
model.add(keras.layers.Dense(128, activation="relu"))

# 4th Layer ( Drop out layer)
model.add(keras.layers.Dropout(0.5))

# Final output Layer
model.add(keras.layers.Dense(1))  # output layer for regression


model.summary()  # print the model summary
model.compile(optimizer="adam",
              loss="mae",
              metrics =[keras.metrics.RootMeanSquaredError()])  # compile the model


training = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prep the test data 
test_data = scaled_data[training_data_len - 60:]  # last 60 days of training data
X_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])  # previous 60 days



X_test = np.array(X_test)  # convert to numpy array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1 ))  # reshape for LSTM input

# Make predictions
predictions = model.predict(X_test)  # make predictions on the test data
predictions = scaler.inverse_transform(predictions)  # inverse transform to get actual values

# Plotting data
train = data[:training_data_len]  # training data
test = data[training_data_len:] 

test = test.copy()

test['Predictions'] = predictions  # add predictions to the test data

plt.figure(figsize=(12, 8))
plt.plot(train['date'], train['close'], label='Train (Actual)', color='blue')
plt.plot(test['date'], test['close'], label='Test (Actual)', color='orange')
plt.plot(test['date'], test['Predictions'], label='Predictions', color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

