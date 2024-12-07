# pip install PySimpleGUI

import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load your trained model
model = keras.models.load_model("CNN_model.keras")

# Function to generate a random test signal
def generate_random_signal(sequence_length, num_features):
    return np.random.rand(sequence_length, num_features)

# Function to classify a test signal
def classify_signal(signal, model):
    signal = np.expand_dims(signal, axis=0)  # Add batch dimension
    predictions = model.predict(signal)
    return np.argmax(predictions), predictions

# Create the PySimpleGUI UI layout
layout = [
    [sg.Text("Time Series Classification UI")],
    [sg.Text("Choose Category:"), sg.Combo(["is_fraudulent", "customer_age"], key="category", default_value="is_fraudulent")],
    [sg.Button("Generate Test Signal"), sg.Button("Exit")],
    [sg.Text("Classification Result:"), sg.Text("", size=(20, 1), key="result")],
    [sg.Canvas(key="canvas")],
]

# Create the window
window = sg.Window("Time Series Classification UI", layout, finalize=True)

# Visualization function
def plot_signal(signal, category):
    plt.figure(figsize=(5, 3))
    plt.plot(signal, label=f"Test Signal ({category})")
    plt.legend()
    plt.title("Generated Test Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Event loop for the UI
sequence_length = 50  # Example sequence length
num_features = 10     # Number of features in your dataset

while True:
    event, values = window.read()

    if event in (sg.WINDOW_CLOSED, "Exit"):
        break

    if event == "Generate Test Signal":
        # Generate a random test signal
        category = values["category"]
        test_signal = generate_random_signal(sequence_length, num_features)

        # Classify the test signal
        predicted_class, prediction_probs = classify_signal(test_signal, model)

        # Update the classification result
        result_text = f"Class: {predicted_class}, Probabilities: {prediction_probs}"
        window["result"].update(result_text)

        # Plot the test signal
        plot_signal(test_signal[:, 0], category)

# Close the window
window.close()
