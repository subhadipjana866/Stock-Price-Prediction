from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import  load_model
import math
import datetime as dt
import yfinance as yf
from sklearn.metrics import mean_squared_error
import streamlit as st

with open('style.css')as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

com_dict={"TATASTEEL.NS":"Data/Tatasteel_pred.h5","ADANIGREEN.NS":"Data/Adani_green_enery.h5","BHARTIARTL.NS":"Data/Bharti_Airtel.h5"}


st.write("This software currently support all NIFTY50 and BANKNIFTY stocks for prediction.")

company = st.text_input("Enter Stock Ticker","TATASTEEL.NS")
curr_time = dt.datetime.now()
hour = curr_time.hour + 60
minutes = curr_time.minute
time = hour+minutes

if(time <= 930):
  today = dt.datetime.today()
  end = today - dt.timedelta(days = 1)
elif(time >= 930):
  end = dt.datetime.today().strftime("%Y-%m-%d")




start = dt.datetime(2010,1,1)
yf.pdr_override()
data = pdr.get_data_yahoo(company, start, end)

st.subheader(f"Data from previous 5 days")
st.write(data.tail())

scaler=MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(data['Close'].values.reshape(-1,1))
train_data = df1

def create_dataset(dataset, time_step):
  datax,datay = [],[]
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step),0]
    datax.append(a)
    datay.append(dataset[i+time_step,0])
  return np.array(datax),np.array(datay)

time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)



model = load_model(com_dict[company])

train_predict = model.predict(x_train)
train_predict = scaler.inverse_transform(train_predict)
math.sqrt(mean_squared_error(y_train,train_predict))


look_back=time_step

x_input= train_data[len(train_data)-look_back:].reshape(1,-1)
temp_input = list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0

st.subheader("Prediction Days")

prediction_days = int(st.slider("Specify The Number of Days",0,20,5))


while(i<prediction_days):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    
output_price = scaler.inverse_transform(lst_output)
xy =[]
for i in range(prediction_days):
  if (dt.datetime.today().weekday() != 6)|(dt.datetime.today().weekday() != 5):
    xy.append((today + dt.timedelta(days = i)))

st.subheader("Predicted Price")
fig = plt.figure(figsize = (12,6))
plt.rcParams['text.color'] = 'black'
plt.plot(xy,output_price,label='Predicted Price', color='green')
plt.xlabel('Days')
plt.ylabel(f'Predicted {company} Stock Price')
plt.legend()
st.pyplot(fig)