import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the dataset
df = pd.read_csv("balanced_dataset.csv")  # Replace with the path to your dataset

# Load the logistic regression model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Heart Disease Prediction")
st.write("Enter your health parameters to predict the likelihood of heart disease.")

# Input fields with explanations

# BMI
st.markdown("### BMI")
st.write("Body Mass Index (BMI) is a measure of body fat based on height and weight. The standard is etween 18.5 and 24.9")
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
PhysicalHealth = st.number_input(
    "Physical Health (days)", 
    min_value=1,  # Minimum value for input
    max_value=30,  # Maximum value for input
    step=1,  
    value=max(1, int(df["PhysicalHealth"].mean()))  # Ensure the value is at least 1
)

# Mental Health
st.markdown("### Mental Health")
st.write("Number of days in the past 30 days your mental health was not good.")
MentalHealth = st.number_input(
    "Mental Health (days)", 
    min_value=1,  # Minimum value for input
    max_value=30,  # Maximum value for input
    step=1,
    value=max(1, int(df["MentalHealth"].mean()))  # Ensure the value is at least 1
)


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
AgeCategory = st.number_input(
    "Age Category", 
    value=min(10, max(1, int(df["AgeCategory"].mean()))),  # Ensure value is between 1 and 10
    min_value=1,  # Minimum value
    max_value=100,  # Maximum value
    step=1  # Ensure only whole numbers are allowed
)

# Race
st.markdown("### Race")
st.write("Select your race/ethnicity from the options below.")
race_options = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic']
race_mapping = {
    'White': 0,
    'Black': 1,
    'Asian': 2,
    'American Indian/Alaskan Native': 3,
    'Other': 4,
    'Hispanic': 5
}
selected_race = st.selectbox("Race/Ethnicity", race_options)  # User selects from dropdown
Race = race_mapping[selected_race]  # Map the selection to numeric value


# Convert inputs to numerical values before prediction
input_data = np.array([[  # Input data as a 2D array
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

# Add missing features (defaulting to 0 for simplicity)
missing_features = np.zeros((1, 7))  # Create a 2D array with shape (1, 7)
input_data = np.hstack([input_data, missing_features])  # Combine input_data with missing features

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    # Display probability of heart disease
    st.subheader("Results")
    st.write(f"Probability of Heart Disease: {prediction_proba * 100:.2f}%")

    # Display custom message based on prediction
    if prediction == 1:  # Positive prediction
        st.write("It might be very difficult to accept but... I hope you will never ever have a stroke attack and this model only makes fun for you :)")
    else:  # Negative prediction
        st.write("Congratulations! You are a lucky one.")
