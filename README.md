# NeuralNetwork

A 2-hidden layered Neural Nework is implemented from scratch, without using any external machine learning libraries. The neural network is trained on the Breast Cancer dataset from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/).

Before running the neural network, the data is pre-processed, which includes converting categorical to numerical attributes, handle null or missing values, standardization and normalization.

The neural network is trained using different activation functions such as sigmoid, tanh & ReLU. Training and testing error sum for different activation functions is calculated and compared to get the best activation function on the given dataset. 

Hyperparameters used: Learning rate, Max iterations, Train-test data percentage split

For more information on the final results, please take a look at "Assignment 2 Programming Part Report.pdf".

# Steps to run the file

1.) Install Python3 and add scikit-learn, pandas, numpy dependencies
2.) Run the NeuralNet.py file on IDE such as Pycharm. It will ask for user input.
3.) Press the following keys for the activation functions: 
		Press 1 for Sigmoid 
		Press 2 for Tanh 
		Press 3 for ReLu 
		Pressing any other key will result in the activation function being sigmoid
4.) To change the train test split percentage, input the amount in between 0.0 to 1.0 on line 234 for the variable train_test_split_size.
5.) To change the max iterations for training, input the amount on line 237 for the variable max_iteratons.
6.) To change the learning rate for training, input the amount on line 240 for the variable learning_rate.
7.) To change the dataset, either input the file path (relative or absolute) or give the dataset URL on line 244 for the variable dataset_file.

# Sample Output

Training on 90.0% data and testing on 10.0% data using the activation function as relu
After 2000 iterations, and having learning rate as 0.05, the total error is 126.18814749780508
The final weight vectors are (starting from input to output layers)
[[ 0.61667749 -0.26324629  0.21841398 -0.93030455]
 [-0.29084554 -0.84296007  0.3863704  -0.97457467]
 [-0.08090943  0.92263452 -0.33162956 -0.05583316]
 [-0.7892175   0.00615181  0.77137972  0.06875468]
 [-0.43704647 -0.29083062  0.7925605  -0.51702172]
 [-0.95223181  0.93145361 -0.1400642  -0.30742297]
 [ 0.15413526 -0.74694768  0.90008662 -0.37277834]
 [ 0.90566225 -0.56321354 -0.5025996   0.72760359]
 [-0.52946334  0.63020011  0.08779447 -0.61747946]]
[[ 0.17767988 -0.90372641]
 [-0.96825404 -0.90057797]
 [-0.20116942  0.15380792]
 [ 0.73503491  0.57232448]]
[[-0.49059449]
 [-0.83549019]]
Testing error sum using activation function as relu: 16.811852502194906
