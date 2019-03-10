from __future__ import division
import numpy as np
import pandas as pd

"""
The goal of this exercise is to get acquainted with the machine learning building blocks.
There are many methods and algorithms to train a machine learning model, but most of them share the same structure:
1. Load the data and preprocess it
2. Define the hypothesis and cost function
3. Design a learning algorithm (e.g. gradient descent)
4. Set the training hyperparameters (learning rate, number of iterations/epochs, etc.)
5. String all of the above to perform training
6. Evaluate your model and repeat (next week)
"""

# import the packages you plan on using

# predict function that multipy teta vector of the weights wih the X Martix


def hypothesis(X, theta):
    """
    :param X: (numpy array) Set of samples you apply the model on
    :param theta: (numpy array) Model parameters
    :return H: (numpy array or scalar) Model's output on the sample set
    """
    X['bias']=1
    return np.dot(X,theta.reshape(-1, 1))


def cost(X, theta, y):
    """
    :param X: (numpy array) Set of samples you apply the model on
    :param theta: (numpy array) Model parameters
    :param y: (numpy array) Target vector (ground truth for each sample)
    :return cost: (scalar) The model's parameters mean loss for all samples
    """
    m = len(X)
    sose=np.square(hypothesis(X,theta)-y).sum()
    cost=sose/(2*m)
    return cost


def gradient(X, theta, y):
    """
    :param X: (numpy array) Set of samples you apply the model on
    :param theta: (numpy array) Model parameters
    :param y: (numpy array) Target vector (ground truth for each sample)
    :return gradient: (numpy array or scalar) parameter gradients for current step
    """
    error=hypothesis(X,theta)-y.reshape(-1,1)
    grad=X.T

def batch_gradient_descent(X, y, theta, lr, batch_size, num_epochs):
    """
    :param X: (numpy array) rows: examples ,  cols: features
    :param y: (numpy array) target vector
    :param theta: (numpy array) model parameters
    :param lr: (float) learning rate
    :param batch_size: (int) training batch size per iteration
    :param num_epochs: (int) total number of passes through the data
    :return theta: (numpy array) optimized model parameters
    """
    # determine how many steps are needed per epoch

    # start a loop over the number of epochs

    # start a loop over the number of steps required to finish an epoch

    # fetch the next batch from the data

    # compute the gradient for this batch

    # update the model parameters according to learning rate

    # print the cost every epoch

    # return the optimized model parameters

    pass


def main():
    # load the WW2 weather data

    # prepare it for training (grab only relevant columns, split to X (samples) and y (target)

    # initialize theta (model parameters)

    # configure training hyperparameters

    # start training
    dt=pd.read_csv("weatherww2.csv")
    # X = dt.iloc[:, :-1]
    # y = dt.iloc[:, -1]
    max_temp=dt['MaxTemp']
    min_temp=dt['MinTemp']
    min_temp_np = min_temp.values
    max_temp_np = max_temp.values
    print(min_temp_np.transpose())
    print(max_temp_np.transpose())
    # print(y)





if __name__ == '__main__':
    main()
