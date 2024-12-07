import tkinter as tk
from tkinter import ttk
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

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

# Function to generate and classify signal
def generate_and_classify():
    # Get the selected category
    category = category_combobox.get()

    # Generate a random test signal
    sequence_length = 50
    num_features = 10
    test_signal = generate_random_signal(sequence_length, num_features)

    # Classify the test signal
    predicted_class, prediction_probs = classify_signal(test_signal, model)

    # Update result label
    result_text = f"Class: {predicted_class}, Probabilities: {prediction_probs}"
    result_label.config(text=result_text)

    # Plot the signal
    plot_signal(test_signal[:, 0], category)

# Function to plot the signal
def plot_signal(signal, category):
    plt.figure(figsize=(5, 3))
    plt.plot(signal, label=f"Test Signal ({category})")
    plt.legend()
    plt.title("Generated Test Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create the Tkinter application
app = tk.Tk()
app.title("Time Series Classification App")

# Dropdown to select category
tk.Label(app, text="Choose Category:").grid(row=0, column=0, padx=10, pady=10)
category_combobox = ttk.Combobox(app, values=["is_fraudulent", "customer_age"])
category_combobox.grid(row=0, column=1, padx=10, pady=10)
category_combobox.set("is_fraudulent")

# Button to generate and classify signal
generate_button = tk.Button(app, text="Generate Test Signal", command=generate_and_classify)
generate_button.grid(row=1, column=0, columnspan=2, pady=10)

# Label to display the classification result
result_label = tk.Label(app, text="Classification Result:", font=("Arial", 12))
result_label.grid(row=2, column=0, columnspan=2, pady=10)

# Run the application
app.mainloop()
