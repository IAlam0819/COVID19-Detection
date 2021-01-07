# COVID19-Detection
In this project, we will be building an image classifier model. We will be dealing with the image data and so, we will be using convolutional neural network, as it gives best results when working with images. Computer doesn’t learn by itself we have to teach the computer and that is done by feeding the images (with labels) to the model and training it. We will be using the images of CT scan of the infected and non-infected people and then building a model that can classify the two images separately.

The dataset is taken from a github repository (https://github.com/UCSD-AI4H/COVID-CT). The COVID-CT-Dataset has 349 CT images containing clinical findings of COVID-19 and 349 Non-COVID CT from 216 patients. But we have taken only 300 of COVID-19 and Non- COVID-19 CT scans for this study. The images are collected from COVID19-related papers from medRxiv, bioRxiv, NEJM, JAMA, Lancet, etc. CTs containing COVID-19 abnormalities are selected by reading the figure captions in the papers. All copyrights of the data belong to the authors and publishers of these papers. 

When dealing with images dataset, we use a set of layers to build the model and they are convolutional layer, pooling layer and fully connected layer. Starting with the Conv2D layers(tf.keras.layers.Conv2D), this layer helps in condensing an input image by passing it to a filter and then using an activation function on the output vector. The arguments of the Conv2D layer are as follows:

•	Filters: it is the number of outputs filters in the convolutions.

•	Kernel size: it denotes the height and width of the 2D convolution window. In our case, it is (2 × 2).

•	Activation function: we will be using the ‘relu’ activation function. Relu means -If x>0 return x, else return 0.

•	Input shape: it mentions the size of the input image, in our case, the input_shape = (224, 224, 3).  Here 3 denotes the red, green and blue channel, if the image is in gray scale then it will be equal to 1.

A Convolutional layer is followed by a pooling layer (tf.keras.layers.MaxPool2D). This layer helps in reducing the size of the image and keeping only the most effective elements of the image vector. If using 3 by 3 pooling layer, it will select 1 pixel out of the 9 pixels, in other words reduce the size of each feature map by a factor of 3.

Next, we'll configure the specifications for model training. We will train our model with the binary_crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. 

#Data Preprocessing

Let's set up data generators that will read pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network. We'll have a generator for the training images,it will yield batches of images of size 224x224 and their labels (binary).

Data that goes into neural networks should be normalized in some way to make it more amenable to processing by the network. We will preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range).

In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches. These generators can then be used with the Keras model methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

#Training

Let's train for 15 epochs.

The Loss and Accuracy are a great indication of progress of training. It's making a guess as to the classification of the training data, and then measuring it against the known label, calculating the result. Accuracy is the portion of correct guesses.
