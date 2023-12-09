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
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of Person")
    
    
    # code for x
    diagnosis = ''
    
    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    if diagnosis == "Person is diabetic":
        st.success(diagnosis)
    else:
        st.error(diagnosis)

if __name__ == "__main__":
    main()


































