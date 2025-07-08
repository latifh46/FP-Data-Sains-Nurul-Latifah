# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Obesity Category Predictor')

# Load pipeline
model = joblib.load('obesity_pipeline.pkl')

# Form input
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age (years)', 10, 90, 25)
height = st.number_input('Height (m)', 1.2, 2.5, 1.70, step=0.01)
weight = st.number_input('Weight (kg)', 30.0, 200.0, 70.0, step=0.1)
family_history = st.selectbox('Family history with overweight', ['yes', 'no'])
favc = st.selectbox('Frequent high-caloric food (FAVC)', ['yes', 'no'])
fcvc = st.slider('Veggies consumption (FCVC)', 1, 3, 2)
ncp  = st.slider('Meals per day (NCP)', 1, 4, 3)
caec = st.selectbox('Snacks (CAEC)', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('SMOKE', ['yes', 'no'])
ch2o = st.slider('Water cups / day (CH2O)', 1, 3, 2)
scc = st.selectbox('Calories monitoring (SCC)', ['yes', 'no'])
faf = st.slider('Physical activity (hrs) FAF', 0.0, 3.0, 1.0, step=0.25)
tue = st.slider('Technology use hrs (TUE)', 0.0, 2.0, 1.0, step=0.25)
calc = st.selectbox('Alcohol (CALC)', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportation', ['Public_Transportation','Walking','Automobile','Motorbike','Bike'])

if st.button('Predict'):
    row = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }])
    prediction = model.predict(row)[0]
    st.success(f'Predicted category: {prediction}')
