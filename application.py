import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned.csv')


st.title('Car Price Prediction')


cities = sorted(car['city'].unique())
car_models = sorted(car['name'].unique())
year = sorted(car['year'].unique(), reverse=True)
fuel_type = car['fuel_type'].unique()

cities.insert(0, 'Select City')

city = st.selectbox('Select City', cities)
car_model = st.selectbox('Car Model', car_models)
year = st.selectbox('Year', year)
fuel_type = st.selectbox('Fuel Type', fuel_type)
driven = st.number_input('Kilometers Driven')

if st.button('Predict'):
    prediction = model.predict(pd.DataFrame(columns=['name', 'kms_driven', 'fuel_type','city', 'year',],
                                             data=np.array([car_model,driven, fuel_type,city, year,]).reshape(1, 5)))
    st.write(f'Predicted Price: {np.round(prediction[0], 2)}Lakhs')
