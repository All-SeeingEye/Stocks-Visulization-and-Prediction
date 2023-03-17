import yfinance as yf
import plotly.graph_objs as plt
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

tsla = yf.Ticker("TTM")

#data = yf.download(tickers='TTM', period='1d', interval='1m')

data_train = yf.download(tickers='TTM', period='5y', interval='1d')


#fig = plt.Figure(data=[plt.Candlestick(x=data.index,open=data['Open'],high=data['High'],low=data['Low'],close=data['Close'], name = 'TESLA Live Market Data')])

#print(data_train)

#fig.show()


sc = MinMaxScaler(feature_range=(0,1)) 
training_set_scaled = sc.fit_transform(data_train)
#print(training_set_scaled)

#print(len(data_train))

X_train = []
y_train = []
for i in range(60, 2035):
    X_train  = np.append(training_set_scaled[i-60:i, 0], X_train) 
    y_train = np.append(training_set_scaled[i-60:i, 0], y_train)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1))

print(X_train.shape)


model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape= (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True)) 
model.add(Dropout(0.2)) 
model.add(LSTM(units=50,return_sequences=True)) 
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) 
model.compile(optimizer='adam',loss='mean_squared_error') 
model.fit(X_train,y_train,epochs=5,batch_size=32)


data_test = yf.download(tickers="TTM", period='1y', interval='1d')
real_stock_price = data_test.iloc[:, 1:2].values


dataset_total = pd.concat((data_train['Open'], data_test['Open']), axis = 0)
#print(len(data_test))

inputs = dataset_total[len(dataset_total) - len(data_test) - 60:].values
print(len(inputs))
inputs = inputs.reshape(-1,1)

inputs = sc.fit_transform(inputs)

X_test = []
for i in range(60, 76):
    X_test= np.append(inputs[i-60:i, 0], X_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 1)) 
    predicted_stock_price = model.predict(X_test) 
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction') 
plt.xlabel('Time')
plt.ylabel('TATA Stock Price') 
plt.legend()
plt.show()