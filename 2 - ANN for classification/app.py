# pip install streamlit

import pickle
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

# Input fields with real words for binary values
# Input fields with explanations

# BMI
st.markdown("### BMI")
st.write("Body Mass Index (BMI) is a measure of body fat based on height and weight.")
BMI = st.number_input("Enter your BMI", value=float(df["BMI"].mean()))

# Smoking
st.markdown("### Smoking")
st.write("Have you smoked at least 100 cigarettes in your entire life?")
Smoking = st.selectbox("Smoking Status", ["No", "Yes"])  # Map: "No" = 0, "Yes" = 1

# Alcohol Drinking
st.markdown("### Alcohol Drinking")
st.write("Do you have more than 14 drinks per week for men or more than 7 drinks per week for women?")
AlcoholDrinking = st.selectbox("Alcohol Consumption", ["No", "Yes"])  # Map: "No" = 0, "Yes" = 1

# Stroke
st.markdown("### Stroke")
st.write("Have you ever been diagnosed with a stroke?")
Stroke = st.selectbox("Stroke History", ["No", "Yes"])  # Map: "No" = 0, "Yes" = 1

# Physical Health
st.markdown("### Physical Health")
st.write("Number of days in the past 30 days your physical health was not good.")
PhysicalHealth = st.number_input("Physical Health (days)", value=float(df["PhysicalHealth"].mean()))

# Mental Health
st.markdown("### Mental Health")
st.write("Number of days in the past 30 days your mental health was not good.")
MentalHealth = st.number_input("Mental Health (days)", value=float(df["MentalHealth"].mean()))

# Difficulty Walking
st.markdown("### Difficulty Walking")
st.write("Do you have difficulty walking or climbing stairs?")
DiffWalking = st.selectbox("Difficulty Walking", ["No", "Yes"])  # Map: "No" = 0, "Yes" = 1

# Sex
st.markdown("### Sex")
st.write("What is your biological sex?")
Sex = st.selectbox("Sex", ["Female", "Male"])  # Map: "Female" = 0, "Male" = 1

# Age Category
st.markdown("### Age Category")
st.write("Age category, where higher values represent older age groups.")
AgeCategory = st.number_input("Age Category", value=float(df["AgeCategory"].mean()))

# Race
st.markdown("### Race")
st.write("Race/ethnicity encoded numerically.")
Race = st.number_input("Race (encoded)", value=float(df["Race"].mean()))

# Convert inputs to numerical values
input_data = np.array([[
    BMI,
    1 if Smoking == "Yes" else 0,
    1 if AlcoholDrinking == "Yes" else 0,
    1 if Stroke == "Yes" else 0,
    PhysicalHealth,
    MentalHealth,
    1 if DiffWalking == "Yes" else 0,
    1 if Sex == "Male" else 0,
    AgeCategory,
    Race
]])

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
