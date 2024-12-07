# simple app to play with random generated signals

# what you can do?
# 1 - choose a category
# 2 - generate a signal
# 3 - display results
# 4 - plot the signal

# simple app to play with random generated signals

# import libraries
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the model from a pickle file
with open("CNN_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)


# function to generate a random test signal
def generate_random_signal(sequence_length, num_features):
    return np.random.rand(sequence_length, num_features)

# function to use the real tensorflow model for predictions
def model_prediction(signal, category):
    # expand dimensions to match the model's input shape
    signal = np.expand_dims(signal, axis=0)  # shape: (1, sequence_length, num_features)

    # use the model to predict
    predictions = model.predict(signal)

    # get the class with the highest probability
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions[0]

# function to generate and classify signal
def generate_and_classify():
    # get the selected category
    category = category_combobox.get()

    # generate a random test signal
    sequence_length = 50
    num_features = 10  # adjust based on your model's input
    test_signal = generate_random_signal(sequence_length, num_features)

    # classify using the real model
    predicted_class, prediction_probs = model_prediction(test_signal, category)

    # update result label
    result_text = f"class: {predicted_class}, probabilities: {prediction_probs}"
    result_label.config(text=result_text)

    # plot the signal
    plot_signal(test_signal[:, 0], category)

# function to plot the signal
def plot_signal(signal, category):
    plt.figure(figsize=(5, 3))
    plt.plot(signal, label=f"test signal ({category})")
    plt.legend()
    plt.title("generated test signal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# create the tkinter application
app = tk.Tk()
app.title("simple time series classification app")

# dropdown to select category
tk.Label(app, text="choose category:").grid(row=0, column=0, padx=10, pady=10)
category_combobox = ttk.Combobox(app, values=["is_fraudulent", "customer_age"])
category_combobox.grid(row=0, column=1, padx=10, pady=10)
category_combobox.set("is_fraudulent")

# button to generate and classify signal
generate_button = tk.Button(app, text="generate test signal", command=generate_and_classify)
generate_button.grid(row=1, column=0, columnspan=2, pady=10)

# label to display the classification result
result_label = tk.Label(app, text="classification result:", font=("Arial", 12))
result_label.grid(row=2, column=0, columnspan=2, pady=10)

# run the application
app.mainloop()
