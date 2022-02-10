from asyncio.windows_events import NULL
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st
from PIL import Image
import os
from keras.models import load_model
from datetime import datetime

st.sidebar.header("Stock Price Prediction")
image = Image.open("ariwells-logo.jpg")
st.sidebar.image(image,width = 200)
#st.sidebar.subheader("Stock Code")
stockCode = st.sidebar.text_input("Stock Code",value="005930.KS")
startDate = st.sidebar.text_input("Start",value='2012-01-01')
endDate = st.sidebar.text_input("End",value='2021-12-28')
st.title ("Stock Price Prediction")
st.header(stockCode)
ok= st.sidebar.button("Stock Code")
df = web.DataReader(stockCode,data_source='yahoo',start= startDate,end=endDate)
graph=plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel("Date",fontsize=18)
plt.ylabel("Close Price",fontsize=18)

if ok:
    st.write("")
    st.table(df.tail(2))
    st.text_area("Query Result",pd.DataFrame.to_string(df),height=300)
    st.pyplot(graph)

def preprocessing():
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset)*.8)
    #st.write(training_data_len)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len,:]
    x_train=[]
    y_train=[]

    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
        if i<=61:
            print(x_train)
            print(y_train)
            print()

    x_train, y_train = np.array(x_train),np.array(y_train)
    #Reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    return (x_train, y_train,training_data_len,scaled_data,dataset,scaler,data)

def training():
    
    x_train,y_train,training_data_len,scaled_data,dataset,scaler,data = preprocessing()
    model = Sequential()

    model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.summary()
    model.fit(x_train,y_train,batch_size=1,epochs=1)
    st.write("...trained completely...")

    target_dir = './models/'
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')
    return NULL

def validating():
    x_train,y_train,training_data_len,scaled_data,dataset,scaler,data = preprocessing()
    test_data = scaled_data[training_data_len-60:,:]
    x_test  = []
    y_test = dataset[training_data_len:,:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    x_test = np.array(x_test)
    x_test= np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    return (x_test,y_test,scaler,training_data_len,data)

    
st.sidebar.subheader("Please train a LSTM model!")
trainOK = st.sidebar.button("Deep Learning")
st.sidebar.subheader("Predict a Stock Price!")
predictionOK = st.sidebar.button("Predict")

if trainOK:
    st.sidebar.write("Please wait a minute.")
    training()
    st.table(df.tail(2))
    st.text_area("Query Result",pd.DataFrame.to_string(df),height=300)
    st.pyplot(graph)

if predictionOK:
    x_test,y_test,scaler,training_data_len,data = validating() 
    model_weights_path = './models/weights.h5'
    model_path = './models/model.h5'
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    #st.write("Validation:")
    #st.write(predictions)
    rmse = np.sqrt(np.mean(predictions-y_test)**2)
    st.write("RMSE:")
    st.write(str(rmse))
    #plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    fig,ax = plt.subplots()
    ax.set_title('Model')
    ax.set_xlabel('Date',fontsize=18)
    ax.set_ylabel('Close Price',fontsize=18)
    ax.plot(train['Close'])
    ax.plot(valid[['Close','Predictions']])
    ax.legend(['Train','Val', 'Predictions'],loc='lower right')
    st.text_area("Comparison between Validation Data and Predictions",pd.DataFrame.to_string(valid),height=300)
    st.pyplot(fig)

    
    endDate = datetime.today().strftime('%Y-%m-%d')

    samsung_quote = web.DataReader('005930.KS', data_source='yahoo',start='2012-01-01',end=endDate)
    new_df = samsung_quote.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test=[]
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)

    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    st.subheader("{} Predicted Stock Price:".format(stockCode))
    #st.write("{} Predicted Stock Price:".format(stockCode))
    st.write(str(pred_price))




