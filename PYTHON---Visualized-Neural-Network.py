import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Function to create and train a neural network
def create_neural_network():
    # Generate some example data
    X = np.random.rand(100, 1)
    y = 3 * X + np.random.randn(100, 1) * 0.1
    
    # Create a Sequential model
    model = Sequential([
        Dense(1, input_shape=(1,), activation='linear')
    ])
    
    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=50)
    
    return model

# Function to visualize the neural network
def visualize_neural_network():
    # Create the GUI window
    window = tk.Tk()
    window.title("Neural Network Visualization")
    
    # Create a canvas to draw the neural network
    canvas = tk.Canvas(window, width=400, height=200)
    canvas.pack()
    
    # Draw input layer
    canvas.create_oval(50, 50, 100, 150, fill="blue")
    
    # Draw output layer
    canvas.create_oval(300, 50, 350, 150, fill="blue")
    
    # Draw connection between input and output layers
    canvas.create_line(100, 100, 300, 100, arrow=tk.LAST)
    
    # Start the GUI event loop
    window.mainloop()

# Create and train the neural network
model = create_neural_network()

# Visualize the neural network
visualize_neural_network()
