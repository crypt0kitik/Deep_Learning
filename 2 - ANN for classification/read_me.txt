ANN for classification

In this project, I used a dataset about heart disease.

The link to the dataset is: https://www.kaggle.com/datasets/mdimran6666/heart-disease-nowadays

The target variable is HeartDisease – yes or no. There are 17 columns with different variables that affect whether a person is prone to having heart disease or may already have it.

Project structure:

PART - 1
I implemented Ydata, DataPrep tools, and performed EDA to clean the data.
I balanced all the data except for the binary variables.
I created 2 different neural networks, changing something in each one.
I created a logistic regression model to compare results and understand what worked better in this case: traditional machine learning or the neural networks I implemented.
Below you can find the changes I made when implementing the neural networks:

PART - 2
Neural network 1: classic one
Neural network 2: classic one + EarlyStop
Logistic regression
Random Forest????

Talking about neural networks,
In the dataset, the target variable appears to be HeartDisease, 
which likely indicates whether a patient has heart disease (Yes or No). 

Comparison of neural networks

| Model               | confusion_matrix         | accuracy_score | roc_auc_score | 
|---------------------|--------------------------|----------------|---------------|
| Neural Network 1    |                          |                |               |
| Neural Network 2    | ...                      | . ..           | ...           |
| Logistic regression | [46127 118] x [3486 158] | 92.78%         | 0.627         |


PART - 3
I developed a simple app based on the streamlit module
For using this you need to be in the same directory, where the app is
To run the app you need to write this in the terminal:
python -m streamlit run app.py
