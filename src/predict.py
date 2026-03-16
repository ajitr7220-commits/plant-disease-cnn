import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

img_size = (224,224)

#load model
model = tf.keras.models.load_model("models/plant_disease_model.h5")

#class labels
class_names = ["Healthy", "Powdery", "Rust"]

# load image
img_path = "data/Test/Test/Rust/82add70df6ab2854.jpg"

img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

#prediction
prediction = model.predict(img_array)

prediction_class = class_names[np.argmax(prediction)]

print("predicted class:", prediction_class)