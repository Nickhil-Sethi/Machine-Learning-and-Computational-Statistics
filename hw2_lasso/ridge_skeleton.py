from __future__ import division 

from pprint import pprint

import sys
import os.path as path
sys.path.insert(0, path.abspath(path.join(__file__ ,"../..")))

import numpy as np

from scipy.optimize import minimize
from hw1_sgd.hw1_skeleton_code import regularized_grad_descent, regularized_batch_gradient_descent_plotter, compute_square_loss

##### global helper functions
def ridge(Lambda):
	def ridge_obj(theta):
		return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N) + Lambda*(numpy.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(Lambda, theta):
	return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N)

def percent_match_supports(w,theta,threshold=1e-1):
	"""returns fraction of supports shared by w, theta 

	if support of w > threshold is same as theta"""
	supp_theta = np.array([1. if abs(t) > 0. else 0. for t in theta])
	supp_w = np.array([1. if abs(e) > 0. else 0. for e in w])
	d = np.dot(supp_theta, supp_w)
	return d/sum(supp_theta)

def percent_match_complement_support(w, theta, threshold=1e-1):
	supp_theta = np.array([1. if abs(t) == 0. else 0. for t in theta])
	supp_w = np.array([1. if abs(e) < threshold else 0. for e in w])
	d = np.dot(supp_theta, supp_w)
	return d/sum(supp_theta)

##### Question 1.1
class LinearSystem(object):
	def __init__(self,m=150,d=40,p=.5,var=1.0):
		"""Produces pairs of variables (X_i , y_i) related by y_i = w * X_i + e

		Parameters
		__________

		m : type int
			number of data points
		d : type int
			dimension of system
		p : type float
			probability of non-zero weight being -1 or 1"""
		
		self.m = m
		self.d = d
		self.p = p
		self.var = var

		self.design_matrix()
		self.theta()
		self.y()

	def design_matrix(self):
		"""Generates data matrix of dimensions (m,d)
		
		Parameters
		__________

		None 

		Returns
		_______

		X : type numpy.array of dimensions (m,d)"""

		self.X = np.random.rand(self.m,self.d)

	def theta(self,fixed_theta=True):
		"""Coefficients for linear system.

		Returns
		_______ 

		Theta : type numpy.array (d,) with only 10 components non-zero."""
		
		self.theta = np.zeros(self.d)
		if fixed_theta:
			for i in range(10):
				if i%2==0:
					self.theta[i] = 10.
				else:
					self.theta[i] = -10.
		else:
			for i in range(10):
				if np.random.rand() < self.p:
					self.theta[i] = 10.
				else:
					self.theta[i] = -10.

	def y(self):
		"""generates outputs y = X.dot(theta) + e, e ~ N(0,1)"""
		self.y = self.X.dot(self.theta) + self.var*np.random.randn(self.m)

	def split(self,t_size=60,valid_size=20):
		"""splits data into train, validation, and test sets.

		Returns
		_______

		train_set : type np.array
			training set 
		valid_set : type np.array
			validation set
		test_set : type np.array
			test set
		"""

		train_set = (self.X[:t_size], self.y[:t_size])
		valid_set = (self.X[t_size:t_size+valid_size+1],self.y[t_size:t_size+valid_size+1])
		test_set = (self.X[valid_size+1:],self.y[valid_size+1:])
		return train_set, valid_set, test_set


##### Question 1.2
def ridge_experiments(train, valid, test, lambdas=(1e-6,1e-4,1e-2,.1)):
	
	print("selecting optimal regularization parameter...\n")
	lambda_losses = regularized_batch_gradient_descent_plotter(train[0], train[1], valid[0], valid[1],lambdas=lambdas)
	lambda_losses = np.array(list(lambda_losses))
	argmin = np.argmin(lambda_losses,axis=0)[1]
	opt_lambda = lambda_losses[argmin][0]

	print("optimal regularization parameter selected {}\n".format(opt_lambda))
	
	w_hist, loss_hist = regularized_grad_descent(train[0],train[1],lambda_reg=opt_lambda)
	w = w_hist[-1]
	loss = loss_hist[-1]

	print("optimal loss {}".format(loss))
	return w

##### Question 2
def soft(a,delta):
	"""soft threshold function, used for update step in lasso algorithm"""
	return np.sign(a)*max(np.abs(a)-delta,0)

def lasso(X,y,lamb,w_init=None,eps=10e-6):
	"""Implements lasso shooting algorithm for regularized linear regression.
	
	Parameters
	__________

	X : numpy.array (n, k)
		design matrix 
	y : numpy.array (n, 1)
		output labels 
	lamb : float
		regularization parameter
	eps : float
		convergence criterion for lasso

	Returns
	_______ 

	w : numpy.array (k, 1)
		coefficients learned from lasso"""
	
	D = X.shape[1]

	# initializing weights
	if w_init is None:
		w = np.linalg.inv(X.T.dot(X) + lamb*np.eye(D)).dot( X.T.dot(y) )
	else:
		# assert type(w_init) is float
		w = w_init

	diff = np.inf
	while diff > eps:
		w_old = w
		for j in range(D):
			a_j = 2*X[:,j].dot(X[:,j])
			c_j = 2*X[:,j].dot(y - X.dot(w) + w[j]*X[:,j])
			if a_j == 0. and c_j == 0.:
				w[j] = 0.
			else:
				w[j] = soft(c_j/a_j,lamb/a_j)
		# TODO : better convergence criterion?
		diff = np.linalg.norm(w-w_old)
	return w

def lambda_lasso_cross_validation(X_train,y_train,X_valid,y_valid,lambdas=(1e-6,1e-4,1e-2,.1),plot_results=False):
	"""Performs hyperparameter selection for lasso regression

	Parameters
	__________ 

	X_train, y_train : type np.arrays 
		training set
	X_valid, y_valid : type np.arrays 
		validation set 
	lambdas : type tuple 
		tuple of lambdas to be validated 
	plot_results : type bool
		if true, plots validation error against lambda

	Returns
	_______ 

	opt_lambda : type np.array 
		optimal hyperparameter lambda"""

	assert type(lambdas) is tuple and lambdas

	min_loss = np.inf
	validation_losses = []
	for lamb in lambdas:
		w = lasso(X_train,y_train,lamb)
		validation_loss = compute_square_loss(X_valid,y_valid,w)
		validation_losses.append((lamb,validation_loss))
		if validation_loss < min_loss:
			min_loss = validation_loss
			opt_lambda,opt_w = lamb, w

	if plot_results:
		plt.plot(np.log(lambdas),validation_losses,'r--')
		plt.show()
		plt.close()

	return opt_lambda, opt_w, min_loss

def homotopy_selection(scaling_factor=.9,eps=1e-6):

	print("generating data...\n")
	L = LinearSystem()
	train, valid, test = L.split()


	X, y = train[0],train[1]
	dim  = X.shape[1]

	# initialize w and lambda to 0 and lambda_max
	w = np.zeros(dim)
	lamb = 2*np.max(X.T.dot(y))

	diff = np.inf
	c = 0
	while diff > eps and c < 1000:
		
		w_old = w
		w = lasso(X,y,lamb,w_init=w)
		print("weights {}".format(w))

		lamb *= scaling_factor
		diff = (w_old-w).dot(w_old-w)
		print(diff)
		c += 1
	
	return w

def lasso_experiments(train, valid, test):
	"""Experiments with lasso regression."""

	print("selecting hyperparameter...\n")
	lamb, w, loss = lambda_lasso_cross_validation(train[0],train[1],valid[0],valid[1])

	print("optimal hyperparameter selected {}, loss {}".format(lamb, loss))
	return w

def feature_correlation():
	pass

if __name__=='__main__':

	print("generating data...\n")
	L = LinearSystem(var=.01)
	train, valid, test = L.split()
	print(L.theta)

	w_r = ridge_experiments(train, valid, test)
	print("\n")
	w_l =lasso_experiments(train, valid, test)	

	print("\n")
	print(percent_match_supports(w_r, L.theta))
	print(percent_match_complement_support(w_r, L.theta))
	print("\n")
	print(percent_match_supports(w_l,L.theta))
	print(percent_match_complement_support(w_l, L.theta))
	
	if False:

		X = numpy.loadtxt("X.txt")
		y = numpy.loadtxt("y.txt")
		(N,D) = X.shape
		w  = numpy.random.rand(D,1)

		for i in range(-5,6):
			Lambda 	= 10**i
			w_opt 	= minimize(ridge(Lambda), w)
			print(Lambda, compute_loss(Lambda, w_opt.x))
