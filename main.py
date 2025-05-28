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
plt.show()

# Plot 2 - Trading Volume ( check for outliers and trends)
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['volume'], label='Volume', color='orange')
plt.title("Stock Volume Over Time")
plt.show()

