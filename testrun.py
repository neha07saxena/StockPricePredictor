# Price Predictor using Recurrent Neural Network

# Part 1 - Data Preprocessing



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values 
#iloc is used to get the right column (here, open price)
#we take col in ramge 1 to 2 as upper range is ignored in python, and we avoid numpy error



# Feature Scaling
#We have normalized our stock data in the range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)



# Creating a data structure with 60 timesteps and 1 output
#Previous 60 days' data will be used as input/ train feature for ith day
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)



# Reshaping
#If we wish to add more than one dimensions for features, 
#we can make the last argument other than 1
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential  #used to create Neural Network object for a sequence of layers
from keras.layers import Dense #To add the Output layer
from keras.layers import LSTM #To add LSTM layers
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#We are using adam optimizer (documentation at https://keras.io/optimizers/)

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 8)

"""
epoch is the number of times we want our data to be forward propagated and
then backpropagated.

Instead of updating weights for every stock price's forward pass, we update 
the weights in batch sizes of 32. For every 32 stock prices,
we update the weights once.

Once we initiate training, it takes around 1 hour for all the epochs to run.
Progressively, with every 10 epochs, we can see that the loss is
decreasing. It started from around 5% and finally became much less.
But too less loss may cause overfitting, where the test data loss becomes more.
"""

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Training set plot
plt.plot(training_set, color= 'red', label = 'Actual Stock values')
plt.title('Original Training set')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Normalized training set plot
plt.plot(training_set_scaled, color= 'red', label = 'Normalized Stock values')
plt.title('Normalized Training set')
plt.xlabel('Time')
plt.ylabel('Normalized Google Stock Price')
plt.legend()
plt.show()

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Mean Absolute Percentage Error (MAPE)
mape =  np.mean(np.abs((real_stock_price - predicted_stock_price) / real_stock_price)) * 100
print('MAPE:')
print(mape)


#Calculating Root Mean Squared Error
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print('RMSE: ')
print(rms)



"""rmse = np.sqrt(((predicted_stock_price - real_stock_price) ** 2).mean())
print('RMSE:')
print(rmse)"""
