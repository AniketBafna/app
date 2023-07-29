# Load the Dependencies
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import json

# Load the Model
pipe = pickle.load(open('pipe.pkl','rb'))

# Load the Dataset
car = pd.read_csv('Car_clean.csv')

# Extract the value from data
company = sorted(car['Car_Company'].unique())
year = sorted(car['Car_Year'].unique())
fuel = sorted(car['Car_Fuel'].unique())
location = sorted(car['Car_Location'].unique())

# Title
st.title(':car: Car Price Prediction :car:')

# Align the buttons
company = st.selectbox('Car Company',company)

Model_ex = sorted(car[car['Car_Company'] == company]['Car_Model'].unique())
model = st.selectbox('Car Model',Model_ex)

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("Manufacturing Year", year)
with col2:
    kms = st.number_input("Total Driven (Kms)")

col3, col4 = st.columns(2)
with col3:
    fuel = st.selectbox("Fuel Type", fuel)
with col4:
    location = st.selectbox("Delivery Location",location)

# Predict button
if st.button('Predict Score'):
    input_df = pd.DataFrame({'Car_Year': [year], 'Car_Kms': [kms],'Car_Model':[model],
                             'Car_Fuel': [fuel],'Car_Location': [location],'Car_Company': [company]})
    result = pipe.predict(input_df)
    st.header("Predicted Car Price :fire: - Rs " + str(int(result[0])))


   