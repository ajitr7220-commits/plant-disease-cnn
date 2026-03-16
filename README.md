# Plant Disease Detection using CNN
Python | TensorFlow | Deep Learning | Image Classification

## Overview

This project uses a **Convolutional Neural Network (CNN)** to classify plant leaf diseases from images.
The model can detect three classes:

* Healthy
* Powdery
* Rust

The model is trained using TensorFlow and Keras.

---

## Dataset

The dataset contains plant leaf images in three classes:

- Healthy
- Powdery
- Rust

Dataset is organized into:
Train / Validation / Test folders.

Due to size limits, the dataset is not included in this repository.

Dataset source: Kaggle Plant Disease Dataset

## Project Structure

```
plant-disease-cnn
│
├── data
│   ├── Train
│   ├── Validation
│   └── Test
│
├── models
│   └── plant_disease_model.h5
│
├── src
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---



## Model Architecture

The CNN model includes:

* Convolution Layers
* MaxPooling Layers
* Flatten Layer
* Dense Layers
* Softmax Output Layer

Input image size:

```
224 x 224 x 3
```

---

## Training

To train the model:

```
python src/train.py
```

The trained model is saved in:

```
models/plant_disease_model.h5
```

---

## Evaluation

To evaluate model performance on the test dataset:

```
python src/evaluate.py
```

Example output:

```
Test Accuracy: ~92%
```

---

## Prediction

To predict disease from a new leaf image:

```
python src/predict.py
```

Example output:

```
Prediction: Rust
```

---

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

Main libraries used:

* TensorFlow
* NumPy
* Pillow

---

## Result

* Training Accuracy: ~99%
* Validation Accuracy: ~93%
* Test Accuracy: ~92%

The model successfully detects plant diseases from leaf images.

---

## Future Improvements

* Add more plant disease classes
* Deploy model using Streamlit
* Build a mobile or web application
