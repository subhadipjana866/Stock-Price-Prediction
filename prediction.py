from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
import math
import datetime as dt
import yfinance as yf
from sklearn.metrics import mean_squared_error

# df = pdr.get_data_tiingo('AAPL', api_key = "4b097e5cac191cdf953eb57bcdc40bcac26daf2b")
# df.to_csv('AAPL.csv')
# df = pd.read_csv('AAPL.csv')
# df1 = df.reset_index()['close']

# scaler=MinMaxScaler(feature_range=(0,1))
# df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
company = 'TATASTEEL.NS'

start = dt.datetime(2010,1,1)
end = dt.datetime.now()
yf.pdr_override()
data = pdr.get_data_yahoo(company, start, end)

scaler=MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(data['Close'].values.reshape(-1,1))

training_size = int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step):
  datax,datay = [],[]
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step),0]
    datax.append(a)
    datay.append(dataset[i+time_step,0])
  return np.array(datax),np.array(datay)

time_step = 100
# x_train, y_train = create_dataset(train_data, time_step)
# x_test,y_test = create_dataset(test_data, time_step)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# model = Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error',optimizer='adam')

# model.fit(x_train,y_train,validation_data =(x_test,y_test),epochs=50,batch_size=32,verbose=1)
# model.save('Tatasteel_pred.h5')
model = load_model('Tatasteel_pred.h5')

# train_predict = model.predict(x_train)
# test_predict = model.predict(x_test)
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# math.sqrt(mean_squared_error(y_train,train_predict))
# math.sqrt(mean_squared_error(y_test,test_predict))

look_back=time_step
# trainPredictPlot = np.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict

# testPredictPlot = np.empty_like(df1)
# testPredictPlot[:, :]=np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :]= test_predict

# plt.plot(scaler.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

x_input= test_data[len(test_data)-look_back:].reshape(1,-1)
temp_input = list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0
days = 5
while(i<5):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
day_new = np.arange(1,look_back+1)
day_pred = np.arange(look_back+1,look_back+1+days)

plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-look_back:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()