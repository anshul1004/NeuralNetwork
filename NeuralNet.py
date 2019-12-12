'''
Authors: Anshul Pardhi, Ashwani Kashyap
'''

#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
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
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, train, train_test_split_size, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train, header=None)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # Split dataset into training set and testing set on the basis of test set size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=train_test_split_size)

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X_train
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X_train), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X_train), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid(x)
        elif activation == "tanh":
            return self.__tanh(x)
        elif activation == "relu":
            return self.__relu(x)
        return None

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #
    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(x)
        elif activation == "tanh":
            self.__tanh_derivative(x)
        elif activation == "relu":
            self.__relu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # tanh function
    def __tanh(self, x):
        return np.tanh(x)

    # derivative of tanh function
    def __tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # ReLu function
    def __relu(self, x):
        return np.maximum(0, x)

    # derivative of Relu function (Assuming a derivative value of 0 for x=0)
    def __relu_derivative(self, x):
        return (x > 0) * 1

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #
    def preprocess(self, X):
        df = X

        #Convert categorical attributes to numerical attributes
        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes.astype('int64')

        arr = df.values

        #Handle null or missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(arr)
        arr = imputer.transform(arr)

        #Standardization, converting mean to 0 and standard deviation to 1
        scaler = StandardScaler().fit(arr)
        arr = scaler.transform(arr)

        df = pd.DataFrame(arr)
        return df

    # Below is the training function
    def train(self, activation="sigmoid", max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X_train, activation)
            error = 0.5 * np.power((out - self.y_train), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)
            self.w23 = self.w23 + update_layer2
            self.w12 = self.w12 + update_layer1
            self.w01 = self.w01 + update_input
        print("After " + str(max_iterations) + " iterations, and having learning rate as " + str(learning_rate) + ", the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, input, activation="sigmoid"):
        # pass our inputs through our neural network
        in1 = np.dot(input, self.w01)
        self.X12 = self.__activation(in1, activation)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3, activation)
        return out

    def backward_pass(self, out, activation="sigmoid"):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions
    def compute_output_delta(self, out, activation="sigmoid"):
        diff = self.y_train - out
        delta_output = None
        if activation == "sigmoid":
            delta_output = diff * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = diff * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = diff * (self.__relu_derivative(out))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions
    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        prod = self.deltaOut.dot(self.w23.T)
        delta_hidden_layer2 = None
        if activation == "sigmoid":
            delta_hidden_layer2 = prod * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = prod * (self.__tanh_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = prod * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions
    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        prod = self.delta23.dot(self.w12.T)
        delta_hidden_layer1 = None
        if activation == "sigmoid":
            delta_hidden_layer1 = prod * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = prod * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = prod * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function
    def predict(self, activation="sigmoid", header = True):
        out = self.forward_pass(self.X_test, activation)
        error = 0.5 * np.power((out - self.y_test), 2)
        return np.sum(error)

if __name__ == "__main__":
    inp = input("Press the following keys for the activation functions: \n Press 1 for Sigmoid \n Press 2 for Tanh \n Press 3 for ReLu \n Pressing any other key will result in the activation function being sigmoid")
    if inp=="2":
        activation = "tanh"
    elif inp=="3":
        activation = "relu"
    else:
        activation = "sigmoid"

    #Specify the train_test split. The value of train_test_split_size indicates that much % of testing data and remaining % of training data
    train_test_split_size=0.10

    #Specify the maximum number of iterations to train the neural network
    max_iteratons = 2000

    #Specify the learning rate
    learning_rate = 0.05

    #Specify the dataset csv file (Using Breast cancer dataset from UCI by default)
    #Breast Cancer UCI data link: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data
    dataset_file="breast-cancer.csv"

    print("Training on " + str((1-train_test_split_size)*100) + "% data and testing on " + str(train_test_split_size*100) + "% data using the activation function as " + str(activation))

    #Initialize the neural network
    neural_network = NeuralNet(dataset_file, train_test_split_size)

    #Train the neural network
    neural_network.train(activation, max_iteratons, learning_rate)

    #Test the neural network
    testError = neural_network.predict(activation)
    print("Testing error sum using activation function as " + str(activation) + ": " + str(testError))