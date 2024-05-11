#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 01:46:24 2023

@author: bakru_k78
"""

import numpy as np
import pickle
import streamlit as st

# loaded saved model
# loaded_model = pickle.load(open("/home/bakru_k78/VsCodeProject/Final-Year-Project/trained_model.sav", "rb"))
loaded_model = pickle.load(open("trained_model.sav", "rb"))

# function for prediction
def diabetes_prediction(input_data):
    # input_data = (0,137,40,35,168,43.1,2.288,33)

    # change input to numpy array
    id_np_arr = np.asarray(input_data)

    # reshape array as we are predicting for one instance
    id_reshaped = id_np_arr.reshape(1, -1)

    # Standardized as we have done to model
    # id_std = scaler.transform(id_reshaped)  

    predict = loaded_model.predict(id_reshaped)

    if predict[0] == 0:
        return "Person is not diabetic"
    else:
        return "Person is diabetic"

def main():
    
    # Giving a title
    st.title("Diabetes Prediction Web App")
    
    # getting input data from user
 	# Pregnancies, Glucose, BloodPressure, SkinThickness,	Insulin, BMI, DiabetesPedigreeFunction, Age 	
    
    # Dropdown menu for gender selection
    gender = st.selectbox("Select Gender", ["Male", "Female"])

    Pregnancies = 0
    # Set number of pregnancies based on gender selection
    if gender == "Male":
        Pregnancies = 0
    else:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=1000, step=1)

    Glucose = st.number_input("Glucose Level (0 to 200)", min_value=0, max_value=200, step=1)
    BloodPressure = st.number_input("Blood Pressure Value (0 to 200)", min_value=0, max_value=200, step=1)
    SkinThickness = st.number_input("Skin Thickness Value (0 to 100)", min_value=0, max_value=100, step=1)
    Insulin = st.number_input("Insulin Level (0 to 500)", min_value=0, max_value=500, step=1)

    # BMI
    # BMI = st.text_input("BMI Value")

    # Get weight and height inputs from user
    weight = st.number_input("Enter Weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Enter Height (m)", min_value=0.0, step=0.01)
    BMI = 0

    # Calculate BMI
    if weight > 0 and height > 0:
        BMI = weight / (height ** 2)

    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value (0 to 5)", min_value=0.00, max_value=5.00, step=0.01)
    Age = st.number_input("Age of Person (0 to 120)", min_value=0, max_value=120, step=1)
    
    
    # code for x
    diagnosis = ''
    
    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    if diagnosis == "Person is diabetic":
        st.success(diagnosis)
    elif diagnosis == "Person is not diabetic":
        st.error(diagnosis)

if __name__ == "__main__":
    main()

