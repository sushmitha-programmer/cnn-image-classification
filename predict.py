import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("model/cnn_model.h5")

# Load image for prediction
img = image.load_img("sample.jpg", target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: Class 1")
else:
    print("Prediction: Class 0")