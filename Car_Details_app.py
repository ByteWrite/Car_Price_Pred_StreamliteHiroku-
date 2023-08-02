import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
import pickle
from sklearn import *
import streamlit as st
import sys

# Load the model and dataset
model = pickle.load(open('lr_model.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

st.title('Car Price Prediction')
st.header('Fill the details to predict car Price')




Brand=st.selectbox('Brand',df['Brand'].unique())
year=st.selectbox('year is',[2017, 2012, 2015, 2014, 2013, 2018, 2016])
km_driven=st.selectbox('km_driven',df['km_driven'].unique())
fuel=st.selectbox('fuel',['Petrol','Diesel','CNG','LPG','Electric'])
seller_type=st.selectbox('seller_type',df['seller_type'].unique())
transmission=st.selectbox('transmission',df['transmission'].unique())
owner=st.selectbox('owner',df['owner'].unique())


if st.button('Predict Laptop Price'):
        

        pred = model.predict([[Brand,year,km_driven,fuel,seller_type,transmission,owner]])
        output = round(pred[0],2)
        if pred < 0: # handeling negative outputs.
            st.error('The input values must be irrelevant, try again by giving relevent information.')
      
        write = str(pred) 
        st.success(write)







