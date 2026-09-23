# Object Detection and Image Recognition

## 📌 Overview

This repository contains Artificial Intelligence and Computer Vision practical work completed as part of my academic coursework.

The practical exercises provided hands-on exposure to image-based machine learning using Python, TensorFlow, Keras, OpenCV and NumPy.

The repository currently includes a Dog and Cat image detection/classification notebook and a handwritten digit recognition program.

---

## 🎯 Objectives

- Understand the basic workflow of image-based machine learning.
- Work with image datasets for classification.
- Train simple neural-network models using TensorFlow and Keras.
- Perform basic image preprocessing using OpenCV and NumPy.
- Explore how trained models can be used to recognize images.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Jupyter Notebook

---

## 🐶🐱 Dog and Cat Image Detection

The notebook contains an image classification exercise for distinguishing between dog and cat images.

The workflow includes:

- Preparing image data
- Organizing images into classes
- Training a neural-network based model
- Evaluating the model
- Using the trained model for image recognition

The complete implementation is available in:

`Dog and Cat image detector model.ipynb`

---

## 🔢 Digit Recognition

The repository also contains a handwritten digit recognition exercise using the MNIST dataset.

The implementation:

- Loads the MNIST dataset.
- Normalizes the image data.
- Builds a simple neural network using TensorFlow/Keras.
- Trains the model for digit classification.
- Provides an OpenCV drawing canvas.
- Processes the drawn digit and predicts its numerical value.

The model uses a `Flatten` layer, a dense hidden layer with ReLU activation, and a 10-class softmax output layer.

Implementation:

`digit recognizer (0-9).py`

---

## 🔄 Basic Workflow

```text
Image Dataset
      ↓
Data Preprocessing
      ↓
Model Building
      ↓
Model Training
      ↓
Model Prediction
      ↓
Recognized Class
