from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model = load_model(r'C:\Users\amit\OneDrive\Desktop\MAJOR PROJECT PLANT HEALTH\plant_health_analysis\plant_model.h5')

# Access training history if available (only works if model was saved with history)
try:
    history = model.history.history
except AttributeError:
    print("Model does not contain training history. You need the 'history' object returned by model.fit().")

# If you have the history separately (from training), use this to plot:
def plot_accuracy(history):
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage (replace with your real history dictionary)
# plot_accuracy(history)
