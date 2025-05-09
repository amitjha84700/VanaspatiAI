from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model(r'C:\\Users\\amit\\OneDrive\\Desktop\\MAJOR PROJECT PLANT HEALTH\\plant_health_analysis\\plant_model.h5')

# Compile the model (required for training/evaluation)
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Image data generator with validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Define image target size (from your model summary)
target_size = (256, 256)

# Training data generator
train_generator = datagen.flow_from_directory(
    r'C:\Users\amit\OneDrive\Documents\python_code\website\websites\plant health analysis\plant-health-analysis-website-main\train',        # <-- Replace with your dataset path
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    r'C:\Users\amit\OneDrive\Documents\python_code\website\websites\plant health analysis\plant-health-analysis-website-main\valid',        # <-- Replace with your dataset path
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Fit the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
