# NumberClassifier
A Java implementation of a neural network which can predict hand drawn numbers from 0 to 9 wrapped in an easy to use GUI. The application continually makes predictions about the number you are drawing so you can follow the "thought"-process of the neural network. 

The neural network is trained on the MNIST data set, which can be downloaded [here](https://pjreddie.com/projects/mnist-in-csv/). If you wish to train a new neural network, place the data sets into a new folder called mnist or alter the file names in the loadData function in NNTrainer.java.

## GUI:
Main.java handles the representation and styling of the GUI. All user input is also mediated by this class.

## MNIST:
### MNIST.java
Dummy class which contains often used variables from the MNIST data set - i.e. image sizes.

### MNISTLoader.java
Class for loading data from the MNIST files, normalizing the data and turning it into the right format for the neural network.

### NNTrainer.java
A class for training and testing neural networks on the MNIST data set. Requires the MNIST and MNISTLoader classes.


## Neural Network:
The neural network used contains 240 hidden nodes and correctly classifies 94.96% of the images in the MNIST data set.
### NeuralNetwork.java
The NeuralNetwork class is used to make guesses about what numbers are drawn on the canvas.
It is implemented such that it only dependent on the cern-libary, which means it can be easily used for other project as well.
The class contains a variety of functionalities including:
- Saving the weights and biases to a file.
- Loading weights and biases from a file.
- Forward and back propagation.
- An easy-to-use test method.

## Motivation for creating the application:
I made the application because I wanted to learn about neural networks and how image recognition works. 
I thought it would be boring to just analyze data without any user interaction, so this project seemed like a good mix between user interaction and neural networks. It should be said that I also got a lot of inspiration from [TheCodingTrain](https://www.youtube.com/thecodingtrain).

# What i have learned
- How to implement neural networks and the math behind it.
- How to preprocess image data.
- How to convert pixel data from a canvas into a matrix.


# What the application looks like
![](GUI_img.png)
