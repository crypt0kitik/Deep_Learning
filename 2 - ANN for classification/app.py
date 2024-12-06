{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Streamlit UI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install streamlit\n",
    "\n",
    "# import libraries\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle \n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv(\"balanced_dataset.csv\")  # Replace with your dataset file path\n",
    "\n",
    "# Split data into features (X) and target (y)\n",
    "X = df.drop(columns=[\"HeartDisease\"])\n",
    "y = df[\"HeartDisease\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load a pre-trained model\n",
    "model = pickle.load(open(\"logistic_model.pkl\", \"rb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-30 10:30:29.320 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:30:29.617 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\e1003118\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python39\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-11-30 10:30:29.623 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:30:29.624 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:30:29.624 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:30:29.624 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:30:29.624 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Streamlit UI\n",
    "st.title(\"Heart Disease Prediction App\")\n",
    "st.write(\"Enter your health metrics below to check the likelihood of heart disease.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-30 10:31:17.939 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.940 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.940 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.941 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.941 Session state does not function when running a script without `streamlit run`\n",
      "2024-11-30 10:31:17.942 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.942 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.944 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.945 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.946 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.947 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.948 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.950 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.951 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.951 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.952 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.954 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.955 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.956 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.957 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.960 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.961 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.962 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.972 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.973 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.974 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.974 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.976 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.977 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.979 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.980 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.981 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.982 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.983 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.984 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.985 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.986 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.986 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.988 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.988 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.989 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.990 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.991 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.993 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.994 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:17.997 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:18.002 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:18.003 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:18.007 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:18.008 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:18.008 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Input fields for each feature in the dataset\n",
    "BMI = st.number_input(\"BMI\", value=float(df[\"BMI\"].mean()))\n",
    "Smoking = st.selectbox(\"Smoking\", [0, 1])  # Assuming 0 = No, 1 = Yes\n",
    "AlcoholDrinking = st.selectbox(\"Alcohol Drinking\", [0, 1])  # Assuming 0 = No, 1 = Yes\n",
    "Stroke = st.selectbox(\"Stroke\", [0, 1])  # Assuming 0 = No, 1 = Yes\n",
    "PhysicalHealth = st.number_input(\"Physical Health (days)\", value=float(df[\"PhysicalHealth\"].mean()))\n",
    "MentalHealth = st.number_input(\"Mental Health (days)\", value=float(df[\"MentalHealth\"].mean()))\n",
    "DiffWalking = st.selectbox(\"Difficulty Walking\", [0, 1])  # Assuming 0 = No, 1 = Yes\n",
    "Sex = st.selectbox(\"Sex\", [0, 1])  # Assuming 0 = Female, 1 = Male\n",
    "AgeCategory = st.number_input(\"Age Category\", value=float(df[\"AgeCategory\"].mean()))\n",
    "Race = st.number_input(\"Race (encoded)\", value=float(df[\"Race\"].mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-30 10:31:25.981 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:25.982 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:25.983 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:25.985 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:25.986 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Prediction button\n",
    "if st.button(\"Predict\"):\n",
    "    # Create an input array for the model\n",
    "    input_data = np.array([[BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth,\n",
    "                            MentalHealth, DiffWalking, Sex, AgeCategory, Race]])\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    prediction_proba = model.predict_proba(input_data)[0][1]\n",
    "\n",
    "    # Display results\n",
    "    st.subheader(\"Results\")\n",
    "    st.write(f\"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}\")\n",
    "    st.write(f\"Probability of Heart Disease: {prediction_proba * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-30 10:31:33.517 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:33.518 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:33.518 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:33.519 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-30 10:31:33.519 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Show dataset summary (optional)\n",
    "if st.checkbox(\"Show Dataset Summary\"):\n",
    "    st.write(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
