# pip install streamlit


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("balanced_dataset.csv")  # Replace with the path to your dataset

# Features and target
X = df.drop(columns=["HeartDisease"])  # Drop the target column
y = df["HeartDisease"]  # Target column

# Load the logistic regression model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Heart Disease Prediction")
st.write("Enter your health parameters to predict the likelihood of heart disease.")

# Input fields for each feature in the dataset
BMI = st.number_input("BMI", value=float(df["BMI"].mean()))
Smoking = st.selectbox("Smoking", [0, 1])  # Assuming 0 = No, 1 = Yes
AlcoholDrinking = st.selectbox("Alcohol Drinking", [0, 1])  # Assuming 0 = No, 1 = Yes
Stroke = st.selectbox("Stroke", [0, 1])  # Assuming 0 = No, 1 = Yes
PhysicalHealth = st.number_input("Physical Health (days)", value=float(df["PhysicalHealth"].mean()))
MentalHealth = st.number_input("Mental Health (days)", value=float(df["MentalHealth"].mean()))
DiffWalking = st.selectbox("Difficulty Walking", [0, 1])  # Assuming 0 = No, 1 = Yes
Sex = st.selectbox("Sex", [0, 1])  # Assuming 0 = Female, 1 = Male
AgeCategory = st.number_input("Age Category", value=float(df["AgeCategory"].mean()))
Race = st.number_input("Race (encoded)", value=float(df["Race"].mean()))

# Prediction button
if st.button("Predict"):
    # Create an input array for the model
    input_data = np.array([[BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth,
                            MentalHealth, DiffWalking, Sex, AgeCategory, Race]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("Results")
    st.write(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
    st.write(f"Probability of Heart Disease: {prediction_proba * 100:.2f}%")

# Show dataset summary (optional)
if st.checkbox("Show Dataset Summary"):
    st.write(df.describe())
