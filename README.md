# Neural-Network
This project implements a 2-hidden-layer neural network. Following are the information useful to understand the project and the output.

1> We are using 3 activation function, ReLU, sigmoid and tanh. This program print the outputs of all three activation functions.

2> We have used the following dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
We took the iris dataset. The .data file can be found at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data .
We have hardcoded this file into our code. We are now passing it to the constructor of the class NeuralNet and inside the __init__ function, we are splitting this dataset into training and test datasets after pre-processing.

3> Even though we have tried with multiple sets of different hyper-parameters, we will use the following set, as it produces the overall better results for all three activation functions than that of other hyper-parameter sets.
Max iteration: 50000
Learning rate: 0.007
Number of neurons in the first hidden layer: 10
Number of neurons in the second hidden layer: 5

4> Since, all the hyper-parameters are fixed, there is no need to pass any of them as command line arguments. Just running the python file should produce output. To find out results for different sets of hyper-parameters, these values need to be modified in the code.
