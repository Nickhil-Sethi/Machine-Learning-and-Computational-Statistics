import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

#######################################
####Q2.1: Normalization

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    (N,p) = np.shape(train)
    mins  = np.amin(train,axis=0)
    maxs  = np.amax(train,axis=0) + mins
    train = (train + mins)/maxs
    test  = (test  + mins)/maxs
    return train, test

########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    N = np.shape(X)[0]
    e = y - X.dot(theta)
    loss = (1/(2*np.float(N)))*e.dot(e)
    return loss


########################################
###Q2.2b: compute the gradient of square loss function

def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    (N,p) = np.shape(X)
    grad = -(1/np.float(N))*np.array([(y - X.dot(theta))*X[:,i] for i in xrange(p)])
    return np.sum(grad,axis=1)

    
#############################################
###Q2.3a: Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features  = theta.shape[0]
    
    e  = np.eye(num_features)
    denominator = np.float(2*epsilon)
    numerator  = np.array([ compute_square_loss(X_train,y_train,theta+epsilon*e[i]) - compute_square_loss(X_train,y_train,theta-epsilon*e[i]) for i in xrange(num_features) ] )
    diff = (true_gradient - numerator/denominator)
    
    return (diff.dot(diff) < tolerance)

#################################################
###Q2.3b: Generic Gradient Checker

def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO

#################################################
####Q2.4a: Batch Gradient Descent

def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X               - the feature vector, 2D numpy array of size (num_instances, num_features)
        y               - the label vector, 1D numpy array of size (num_instances)
        alpha           - step size in gradient descent
        num_iter        - number of iterations to run 
        check_gradient  - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist      - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist       - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist  = np.zeros(num_iter+1) #initialize loss_hist
    theta  = np.ones(num_features) #initialize theta

    count  = 0
    while count < num_iter+1:
        if check_gradient:
            assert grad_checker(X,y,theta)

        grad = compute_square_loss_gradient(X,y,theta)
        theta -= alpha*grad
        theta_hist[count] = theta
        loss_hist[count] = compute_square_loss(X,y,theta)
        count += 1
    
    return theta_hist, loss_hist 

###############################################
# batch gradient descient plotter
def batch_gradient_descent_plotter(X,y,alphas):
    
    losses = []
    alphas.sort()
    for alpha in alphas:
        thetas, loss = batch_grad_descent(X,y,alpha)
        losses.append(loss[-1])

    plt.plot(np.log(alphas),losses,'ro')
    plt.show()

    return zip(alphas,losses)

###################################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
    
###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    return compute_square_loss_gradient(X,y,theta) + 2*lambda_reg*theta

###################################################
###Q2.5b: Batch Gradient Descent with regularization term

def regularized_grad_descent(X, y, alpha=0.01, lambda_reg=1e-6, num_iter=1000):
    """
    Args:
        X               - the feature vector, 2D numpy array of size (num_instances, num_features)
        y               - the label vector, 1D numpy array of size (num_instances)
        alpha           - step size in gradient descent
        lambda_reg      - the regularization coefficient
        num_iter        - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features)                       #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))   #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)                    #Initialize loss_hist

    count = 0
    while count < num_iter + 1:
        grad = compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
        theta -= alpha*grad
        
        theta_hist[count] = theta
        loss_hist[count] = compute_square_loss(X,y,theta) + lambda_reg*(theta.dot(theta))
        
        count += 1

    return theta_hist, loss_hist
    
#############################################
##Q2.5c:  Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss
# optimal lambda typically around 10e-7 or 10e-5
def regularized_batch_gradient_descent_plotter(X_train,y_train,X_valid,y_valid,
    lambdas=(1e-6,1e-4,1e-2,1e-1,1.,10.,100.),alpha=.01):

    train_losses = []
    test_losses = []
    lambdas = list(lambdas)
    lambdas.sort()

    for lamb in lambdas:
        thetas, train_loss = regularized_grad_descent(X_train,y_train,alpha,lamb)
        test_loss = compute_square_loss(X_valid,y_valid,thetas[-1])
        train_losses.append(train_loss[-1])
        test_losses.append(test_loss)

    plt.plot(np.log(lambdas),train_losses,'b--')
    plt.plot(np.log(lambdas),test_losses,'r--')
    plt.show()
    plt.close()

    return zip(lambdas,test_losses)

#############################################
###Q2.6a: Stochastic Gradient Descent

def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, checkin=100):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)                              #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances))                #Initialize loss_hist
    epoch = 1
    while epoch < num_iter:
        instance = 1
        while instance < num_instances:
            if alpha == "1/sqrt(t)":
                alpha_0 = .01/np.sqrt(instance)
            elif alpha == "1/t":
                alpha_0 = .01/float(instance)
            else:
                alpha_0 = alpha
            index = np.random.randint(num_instances)
            vec = np.reshape(X[index,:].T,(1,49))
            grad = compute_regularized_square_loss_gradient(vec,y[index],theta,lambda_reg)
            theta = theta - alpha_0*grad
            theta_hist[epoch][instance] = theta
            loss_hist[epoch][instance] = compute_square_loss(vec,y[index],theta)
            instance += 1

        if type(checkin) is int and epoch%checkin==0:
            print "completed training epoch {}...".format(epoch)
        
        epoch += 1

    return theta_hist, loss_hist

################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)

def main(bias=1.):
    print('loading the dataset...')
    df = pd.read_csv(os.getcwd() + '/hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]
    
    print('splitting into train and test...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("scaling features to [0, 1]...\n")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, bias*np.ones((X_train.shape[0], 1))))   # Add bias term
    X_test = np.hstack((X_test,  bias*np.ones((X_test.shape[0], 1))))    # Add bias term
    
    return (X_train, y_train), (X_test,y_test)

def compare(X_train,y_train,X_test,y_test,dialation=10.):
    t,l1 = regularized_grad_descent(X_train,y_train,lambda_reg=1e-6)

    X_train[:,0] *= dialation
    X_test[:,0] *= dialation
    
    t,l2 = regularized_grad_descent(X_train,y_train,lambda_reg=1e-6)

    plt.plot(np.log(l1),'b--')
    plt.plot(np.log(l2),'r--')
    plt.show()
    plt.close()


if __name__ == "__main__":
    (X_train, y_train), (X_test,y_test) = main()
    
        