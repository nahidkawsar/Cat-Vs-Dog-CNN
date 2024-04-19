# Dogs vs Cats Classification using CNN

This repository contains the code for a Convolutional Neural Network (CNN) model trained to classify images of dogs and cats. The model is implemented using TensorFlow and Keras.

## Dataset:
The dataset used for training and testing the model is the Dogs vs Cats dataset, which can be found on Kaggle here. It consists of a large number of images of dogs and cats.

### Getting Started

#### Prerequisites
Make sure you have the following installed:

* Python
* TensorFlow
* Keras
* OpenCV (cv2)
* Matplotlib

### Installation
Clone this repository:

git clone https://github.com/your_username/dogs-vs-cats-cnn.git

Download the Dogs vs Cats dataset from Kaggle and place it in the appropriate directory.
Install the required dependencies:


pip install -r requirements.txt

### Usage
Run the provided Jupyter Notebook dogs_vs_cats_cnn.ipynb to train and evaluate the CNN model.
Test the trained model on your own images by providing the file path to the image in the notebook.

### Model Architecture

The CNN model architecture consists of three convolutional layers followed by max-pooling layers, batch normalization, dropout layers, and fully connected layers. Here's a summary of the model architecture:


Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 254, 254, 32)      896
_________________________________________________________________
batch_normalization (BatchNo (None, 254, 254, 32)      128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 127, 127, 32)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 125, 125, 64)      18496
_________________________________________________________________
batch_normalization_1 (Batch (None, 125, 125, 64)      256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 62, 62, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 60, 60, 128)       73856
_________________________________________________________________
batch_normalization_2 (Batch (None, 60, 60, 128)       512
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 30, 30, 128)       0
_________________________________________________________________
flatten (Flatten)            (None, 115200)            0
_________________________________________________________________
dense (Dense)                (None, 128)               14745728
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65
=================================================================Total params: 14,851,193

Trainable params: 14,850,745

Non-trainable params: 448


## Results
The model achieves an accuracy of 0.9736 on the validation dataset after 10 epochs of training.


## Future Improvements

Implement data augmentation techniques to improve model generalization.
Experiment with different CNN architectures and hyperparameters.
Deploy the model for real-time predictions.

### Credits
* Dogs vs Cats dataset on Kaggle.

https://www.kaggle.com/datasets/salader/dogs-vs-cats

* TensorFlow and Keras for deep learning framework.

Author:H.M Nahid Kawsar

LinkedIn: https://www.linkedin.com/in/h-m-nahid-kawsar-232a86266


