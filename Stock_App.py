from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import  load_model
import math
import datetime as dt
import yfinance as yf
from sklearn.metrics import mean_squared_error
import streamlit as st
import os
import re
st.set_page_config(
    page_title="Stock_App",
    page_icon="🏠",
)
st.sidebar.success("Stock App")
with open('style.css')as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

# code = """<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7367141985992219"
#      crossorigin="anonymous"></script>"""

# a=os.path.dirname(st.__file__)+'/static/index.html'
# with open(a, 'r') as s:
#     data = s.read()
#     if len(re.findall('ca-', data))==0:
#         with open(a, 'w') as ff:
#             newdata=re.sub('<head>','<head>'+code,data)
#             ff.write(newdata)
#         print("success")


com_dict={"Tata Steel":"TATASTEEL.NS","Adani Green Energy":"ADANIGREEN.NS","Airtel":"BHARTIARTL.NS","Bharat Electronics ltd":"BEL.NS","UPL":"UPL.NS","Ultra Tech Cement":"ULTRACEMCO.NS","Tech Mahindra":"TECHM.NS","Tata Consultancy Services":"TCS.NS","Tata Consumer Products":"TATACONSUM.NS","NTPC":"NTPC.NS","Mahindra & Mahindra":"M&M.NS","Kotak Bank":"KOTAKBANK.NS","Indusland Bank":"INDUSLNDDBK.NS","ICICI Bank":"ICICIBANK.NS","HDFC":"HDFC.NS","Coal India":"COALINDIA.NS","AU Bank":"AUBANK.NS","Adani Enterprise":"ADANIENT.NS","Adani Ports":"ADANIPORTS.NS","Apollo Hospital":"APOLLOHOSP.NS","Axis Bank":"AXISBANK.NS","Bajaj Auto":"BAJAJ-AUTO.NS","Britannia":"BRITANNIA.NS","Grasim":"GRASIM.NS","HDFC Bank":"HDFCBANK.NS","HDFC Life Insuarance":"HDFCLIFE.NS","IDFC First Bank":"IDFCFIRSTB.NS","Reliance":"RELIANCE.NS","Sun Pharma":"SUNPHARMA.NS","Bajaj Finance":"BAJFINANCE.NS","HCL Technology":"HCLTECH.NS","Hero":"HEROMOTOCO.NS","Hindustan Uniliver":"HINDUNILVR.NS","ITC":"ITC.NS","Maruti":"MARUTI.NS","ONGC":"ONGC.NS","Panjab National Bank":"PNB.NS","SBI Life":"SBILIFE.NS","Titan":"TITAN.NS","Cipla":"CIPLA.NS","Power Grid Corporation":"POWERGRID.NS","Dr.Reddy's Laboratories":"DRREDDY.NS","Nestle India":"NESTLEIND.NS","BPCL":"BPCL.NS","Larson & Toubro":"LT.NS","Asian Paints":"ASIANPAINT.NS","Infosys":"INFY.NS","Eichar Motors":"EICHERMOT.NS","Hindalco":"HINDALCO.NS","Federal Bank":"FEDERALBNK.NS","Kalyan Jewellers":"KALYANKJIL.NS"}

mod_dict = {"TATASTEEL.NS":"Data/TATASTEEL.h5","ADANIGREEN.NS":"Data/ADANIGREEN.h5","BHARTIARTL.NS":"Data/BHARTIARTL.h5","UPL.NS":"Data/UPL.h5","ULTRACEMCO.NS":"Data/ULTRACEMCO.h5","TECHM.NS":"Data/TECHM.h5","TCS.NS":"Data/TCS.h5","TATACONSUM.NS":"Data/TATACONSUM.h5","NTPC.NS":"Data/NTPC.h5","M&M.NS":"Data/M&M.h5","KOTAKBANK.NS":"Data/KOTAKBANK.h5","INDUSINDDBK.NS":"Data/INDUSINDDBK.h5","ICICIBANK.NS":"Data/ICICIBANK.h5","HDFC.NS":"Data/HDFC.h5","COALINDIA.NS":"Data/COALINDIA.h5","AUBANK.NS":"Data/AUBANK.h5","ADANIENT.NS":"Data/ADANIENT.h5","ADANIPORTS.NS":"Data/ADANIPORTS.h5","APOLLOHOSP.NS":"Data/APOLLOHOSP.h5","AXISBANK.NS":"Data/AXISBANK.h5","BAJAJ-AUTO.NS":"Data/BAJAJ-AUTO.h5","BRITANNIA.NS":"Data/BRITANNIA.h5","GRASIM.NS":"Data/GRASIM.h5","HDFCBANK.NS":"Data/HDFCBANK.h5","HDFCLIFE.NS":"Data/HDFCLIFE.h5","IDFCFIRSTB.NS":"Data/IDFCFIRSTB.h5","RELIANCE.NS":"Data/RELIANCE.h5","SUNPHARMA.NS":"Data/SUNPHARMA.h5","BAJFINANCE.NS":"Data/BAJFINANCE.h5","HCLTECH.NS":"Data/HCLTECH.h5","HEROMOTOCO.NS":"Data/HEROMOTOCO.h5","HINDUNILVR.NS":"Data/HINDUNILVR.h5","ITC.NS":"Data/ITC.h5","MARUTI.NS":"Data/MARUTI.h5","ONGC.NS":"Data/ONGC.h5","PNB.NS":"Data/PNB.h5","SBILIFE.NS":"Data/SBILIFE.h5","TITAN.NS":"Data/TITAN.h5","CIPLA.NS":"Data/CIPLA.h5","POWERGRID.NS":"Data/POWERGRID.h5","DRREDDY.NS":"Data/DRREDDY.h5","NESTLEIND.NS":"Data/NESTLEIND.h5","BPCL.NS":"Data/BPCL.h5","LT.NS":"Data/LT.h5","ASIANPAINT.NS":"Data/ASIANPAINT.h5","INFY.NS":"Data/INFY.h5","EICHERMOT.NS":"Data/EICHERMOT.h5","HINDALCO.NS":"Data/HINDALCO.h5","FEDERALBNK.NS":"Data/FEDERALBNK.h5","KALYANKJIL.NS":"Data/KALYANKJIL.h5","BEL.NS":"Data/BEL.h5"}

st.write("This software currently support all NIFTY50 and BANKNIFTY stocks for prediction.")

option = st.selectbox("Select the company",("Tata Steel","Bharat Electronics ltd","Adani Green Energy","Airtel","UPL","Ultra Tech Cement","Tech Mahindra","Tata Consultancy Services","Tata Consumer Products","NTPC","Mahindra & Mahindra","Kotak Bank","Indusland Bank","ICICI Bank","HDFC Bank","Coal India","AU Bank","Adani Enterprise","Adani Ports","Apollo Hospital","Axis Bank","Bajaj Auto","Britannia","Grasim","HDFC Bank","HDFC Life Insuarance","IDFC First Bank","Reliance","Sun Pharma","Bajaj Finance","HCL Technology","Hero","Hindustan Uniliver","ITC","Maruti","ONGC","Panjab National Bank","SBI Life","Titan","Cipla","Power Grid Corporation","Dr.Reddy's Laboratories","Nestle India","BPCL","Larson & Toubro","Asian Paints","Infosys","Eichar Motors","Hindalco","Federal Bank","Kalyan Jewellers"))


company = com_dict[option]
curr_time = dt.datetime.now()
hour = curr_time.hour + 60
minutes = curr_time.minute
time = hour+minutes

today = dt.datetime.today()
end = today - dt.timedelta(days = 1)

start = dt.datetime(2010,1,1)
yf.pdr_override()
data = pdr.get_data_yahoo(company, start, end)

st.subheader(f"Data from previous 5 days")
st.write(data.tail())

scaler=MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(data['Close'].values.reshape(-1,1))
total_data = df1

training_size = int(len(df1)*0.70)
test_size = int(len(df1)-training_size)
test_data = df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step):
  datax,datay = [],[]
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step),0]
    datax.append(a)
    datay.append(dataset[i+time_step,0])
  return np.array(datax),np.array(datay)

time_step = 100
x_test, y_test = create_dataset(test_data, time_step)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)



model = load_model(mod_dict[company])

test_predict = model.predict(x_test)
test_predict = scaler.inverse_transform(test_predict)

testPredictPlot = np.empty_like(df1)
testPredictPlot[:,:] = np.nan
testPredictPlot[100:len(test_data)-1,:] = test_predict

st.write("Testing the prediction accuracy with test data")
fig1 = plt.figure(figsize = (12,6))
plt.rcParams['text.color'] = 'black'
plt.plot(scaler.inverse_transform(test_data),label="Actual price")
plt.plot(testPredictPlot,label="Predicted Price")
plt.xlabel('Days')
plt.ylabel(f'{company} Stock Price')
plt.legend()
st.pyplot(fig1)

look_back=time_step

x_input= test_data[len(test_data)-look_back:].reshape(1,-1)
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
for i in range(1,prediction_days+1):
   xy.append(f"{i}")

st.subheader("Predicted Price")
fig2 = plt.figure(figsize = (12,6))
plt.rcParams['text.color'] = 'black'
plt.plot(xy,output_price,label='Predicted Price', color='green')
plt.xlabel('Days')
plt.ylabel(f'Predicted {company} Stock Price')
plt.legend()
st.pyplot(fig2)
