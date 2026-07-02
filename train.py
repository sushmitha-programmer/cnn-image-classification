import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import model

# Data preprocessing
train_data = ImageDataGenerator(rescale=1./255)

training_set = train_data.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(
    training_set,
    epochs=10
)

# Save the trained model
model.save("model/cnn_model.h5")

print("Model trained and saved successfully!")