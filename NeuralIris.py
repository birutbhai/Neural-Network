#####################################################################################################################
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, train, header, h1, h2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        train_dataset = self.preprocess(raw_input)
        nrows = len(train_dataset)
        ncols = len(train_dataset[0])
        X = train_dataset[:,0:ncols-1].reshape(nrows, ncols-1)
        y = train_dataset[:,-1].reshape(nrows, 1)
        #print("****X: "+str(self.X)+"****Y:"+str(self.y))
        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y)
        #print("***********"+str(len(self.X))+"****"+str(len(self.X_test)))

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        #print("****"+str(self.w01))
        self.X01 = self.X 
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))


    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)
        if activation == "ReLu":
            self.__relu(self, x)


    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)
        if activation == "ReLu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh(self, x):
        return (np.exp(x)- np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def __tanh_derivative(self, x):
        return 1- x*x
    
    def __relu(self, x):

        return np.maximum(x,0)
    
    def __relu_derivative(self, x):
        return (x > 0) * 1


    def preprocess(self, X):
        ncols = len(X.columns)
        #print (X)
        class_label = X.iloc[:, (ncols-1)].values
        #Encode class label to an integer value
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(["Iris-setosa","Iris-versicolor","Iris-virginica"])
        #print(list(label_encoder.classes_))
        new_class_label = label_encoder.transform(class_label)
        #print("The labels are: "+ str(new_class_label))
        X.iloc[:, (ncols-1)]=new_class_label
        
        # Drop Missing values
        X.dropna()
        
        # Scale the Data
        scaler = StandardScaler()
        scaler.fit(X)
        #Now apply the transformations to the data:
        X = scaler.transform(X)
        #print(X)
        return X

    # Below is the training function

    def train(self, activation, max_iterations, learning_rate):

        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            #print("DeltaOut: "+str(self.deltaOut)+ "and X23 : "+str(self.X23))
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
            
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))

        print("The final weight vectors are (starting from input to output layers)")

        print(self.w01)
        print(self.w12)
        print(self.w23)


    def forward_pass(self, activation):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        if activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
    
        if activation == "ReLu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)    
            
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)



    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation == "ReLu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        self.deltaOut = delta_output



    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))    
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2



    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1


    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "ReLu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        self.delta01 = delta_input_layer


    def predict(self, activation):
        X = self.X_test
        y = self.y_test       
        if activation == "sigmoid":
            in1 = np.dot(X, self.w01 )
            X12 = self.__sigmoid(in1)
            in2 = np.dot(X12, self.w12)
            X23 = self.__sigmoid(in2)
            in3 = np.dot(X23, self.w23)
            out = self.__sigmoid(in3)
        if activation == "tanh":
            in1 = np.dot(X, self.w01 )
            X12 = self.__tanh(in1)
            in2 = np.dot(X12, self.w12)
            X23 = self.__tanh(in2)
            in3 = np.dot(X23, self.w23)
            out = self.__tanh(in3)
    
        if activation == "ReLu":
            in1 = np.dot(X, self.w01)
            X12 = self.__relu(in1)
            in2 = np.dot(X12, self.w12)
            X23 = self.__relu(in2)
            in3 = np.dot(X23, self.w23)
            out = self.__relu(in3)           

        #print ("The output is "+str(out))

        return 0.5 * np.power((out - y), 2)


if __name__ == "__main__":
    activation = ["sigmoid", "tanh", "ReLu"]
    max_iterations = 50000
    learning_rate = 0.007
    h1 = 10
    h2 = 5
    print ("Max iteration: " + str(max_iterations))
    print ("Learning rate: " + str(learning_rate))
    print ("Number of neurons in the first hidden layer: " + str(h1))
    print ("Number of neurons in the second hidden layer: " + str(h2))
    header = False
    for a in activation:
        print("Results with activation function " + a)
        neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header, h1, h2)
        neural_network.train(a, max_iterations, learning_rate)
        testError = neural_network.predict(a)
        print("Test error with the activation function " + a + " : " + str(np.sum(testError)))