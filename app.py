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


def display_main_app():

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
        Pregnancies = st.number_input(
            "Number of Pregnancies", min_value=0, max_value=1000, step=1)

    Glucose = st.number_input(
        "Glucose Level (0 to 200)", min_value=0, max_value=200, step=1)
    BloodPressure = st.number_input(
        "Blood Pressure Value (0 to 200)", min_value=0, max_value=200, step=1)
    SkinThickness = st.number_input(
        "Skin Thickness Value (0 to 100)", min_value=0, max_value=100, step=1)
    Insulin = st.number_input(
        "Insulin Level (0 to 500)", min_value=0, max_value=500, step=1)

    # BMI
    # BMI = st.text_input("BMI Value")

    # Get weight and height inputs from user
    weight = st.number_input("Enter Weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Enter Height (m)", min_value=0.0, step=0.01)
    BMI = 0

    # Calculate BMI
    if weight > 0 and height > 0:
        BMI = weight / (height ** 2)

    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function value (0 to 5)", min_value=0.00, max_value=5.00, step=0.01)
    Age = st.number_input("Age of Person (0 to 120)",
                          min_value=0, max_value=120, step=1)

    # code for x
    diagnosis = ''

    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    if diagnosis == "Person is diabetic":
        st.success(diagnosis)
    elif diagnosis == "Person is not diabetic":
        st.error(diagnosis)


def display_understanding_diabetes_tab():
    # Colorful heading
    st.markdown(
        "<h1 style='color: #FF4B4B;'>Diabetes Prediction Information</h1>",
        unsafe_allow_html=True
    )

    st.subheader("Pregnancies:")
    st.text("This represents the number of pregnancies the individual has had.")

    st.subheader("Glucose Level:")
    st.text("The concentration of glucose in the blood. It is measured in mg/dL.")

    st.subheader("Blood Pressure Value:")
    st.text(
        "The blood pressure of the individual. It is measured in mmHg (millimeters of mercury).")

    st.subheader("Skin Thickness Value:")
    st.text("The thickness of skin folds on the triceps. It is measured in mm.")

    st.subheader("Insulin Level:")
    st.text("The insulin level in the blood. It is measured in units/mL.")

    st.subheader("BMI (Body Mass Index):")
    st.text("BMI is a measure of body fat based on height and weight. It is calculated by dividing weight in kilograms by the square of height in meters. BMI indicates whether an individual is underweight, normal weight, overweight, or obese.")
    st.code("BMI = weight (kg) / (height (m) ** 2)")

    st.subheader("Diabetes Pedigree Function:")
    st.text(
        "A function that scores the likelihood of diabetes based on family history.")
    st.code("DPF = sum of diabetes cases in relatives / total number of relatives")

    st.subheader("Age of Person:")
    st.text("The age of the individual in years.")


def display_about_tab():
    st.header("Welcome to the Diabetes Prediction Web Application! 👋")

    st.markdown(
        """
        This application serves a dual purpose, benefiting both healthcare professionals and users alike.

        **For Doctors:**
        - Quickly obtain preliminary results for diabetes diagnosis.
        - Enhance diagnostic accuracy through a two-step verification process.
        - Streamline testing procedures for improved patient care.

        **For Users:**
        - Conveniently monitor health status from anywhere.
        - Receive instant feedback on likelihood of having diabetes.
        - Empower proactive health management and informed decision-making.

        With its user-friendly interface and powerful predictive capabilities, the Diabetes Prediction Web Application aims to improve healthcare outcomes, enhance patient engagement, and contribute towards a healthier future for all.
        """
    )

    st.divider()

    st.subheader("Data Sources and Methodology")

    st.markdown(
        """
        **Data Sources:**
        The dataset used in this project was obtained from Kaggle. It can be accessed at:
        """
    )
    st.code("https://www.kaggle.com/code/melikedilekci/diabetes-dataset-for-beginners")

    st.markdown(
        """
        **Methodology:**
        The following machine learning algorithms were used in this project:
        - Support Vector Machine (SVM).
        - Random Forest.
        
        **Algorithm Selection:**
        - Support Vector Machine (SVM) algorithm was chosen for its robustness in handling complex datasets and its ability to find optimal decision boundaries. SVM performs well in high-dimensional spaces and is effective in cases where the data may not be linearly separable.
        - Random Forest, on the other hand, was selected for its ensemble learning approach, which combines multiple decision trees to improve prediction accuracy. It is capable of handling large datasets with high dimensionality and is less prone to overfitting compared to individual decision trees.
        
        **Technology Used:**
        This web application was developed using Streamlit, a powerful Python library for creating interactive web applications.
        """
    )


if __name__ == "__main__":
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "Go to", ["Main App", "Understanding Diabetes", "About"])

    if selection == "Main App":
        display_about_tab()
    elif selection == "Understanding Diabetes":
        display_understanding_diabetes_tab()
    elif selection == "About":
        display_main_app()
