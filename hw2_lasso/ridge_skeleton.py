from __future__ import division 

import sys
import os.path as path
sys.path.insert(0, path.abspath(path.join(__file__ ,"../..")))

import numpy as np

from scipy.optimize import minimize
from hw1_sgd.hw1_skeleton_code import regularized_batch_gradient_descent_plotter

# from hw1_sgd import regularized_batch_gradient_descent_plotter

def ridge(Lambda):
	def ridge_obj(theta):
		return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N) + Lambda*(numpy.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(Lambda, theta):
	return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N)

##### Question 1
class LinearSystem(object):
	def __init__(self,m=150,d=75,p=.5):
		"""Produces pairs of variables (X_i , y_i ) related by y_i = w * X_i + e

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
			for i in xrange(10):
				if i%2==0:
					self.theta[i] = 10.
				else:
					self.theta[i] = -10.
		else:
			for i in xrange(10):
				if np.random.rand() < self.p:
					self.theta[i] = 10.
				else:
					self.theta[i] = -10.

	def y(self):
		"""generates outputs y = X.dot(theta) + e, e ~ N(0,1)"""
		self.y = self.X.dot(self.theta) + np.random.randn(self.m)

	def split(self):
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

		train_set = (self.X[:80], self.y[:80])
		valid_set = (self.X[80:101],self.y[80:101])
		test_set = (self.X[101:],self.y[101:])
		return train_set, valid_set, test_set

##### Question 2
def soft(a,delta):
	"""soft threshold function, used for update step in lasso algorithm"""
	return np.sign(a)*max(np.abs(a)-delta,0)

def percent_match_supports(w,theta,threshold=2.):
	"""returns fraction of supports shared by w, theta 

	if support of w > threshold is same as theta"""
	assert w.shape == theta.shape
	z = zip(w,theta)
	supp = [1 if (abs(x) > threshold and abs(y) > 0) or \
			(abs(x) < threshold and abs(y) < 1e-6) else 0 for (x,y) in z ]
	return sum(supp)/len(supp)

def lasso(X,y,lamb,w_init=None,eps=10e-6):
	"""Implements lasso algorithm for regularized linear regression.
	
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
		assert type(w_init) is float
		w = w_init

	diff = np.inf
	while diff > eps:
		w_old = w.copy()
		for j in xrange(D):
			a_j = 2*X[:,j].dot(X[:,j])
			c_j = 2*X[:,j].dot(y - X.dot(w) + w[j]*X[:,j])
			w[j] = soft(c_j/a_j,lamb/a_j)
		diff = np.linalg.norm(w-w_old)
	return w

def main():
	L = LinearSystem()
	train, valid, test = L.split()
	
	lambda_losses = regularized_batch_gradient_descent_plotter(train[0], train[1], valid[0], valid[1])
	lambda_losses = np.array(lambda_losses)
	opt_lambda = lambda_losses[np.argmin(lambda_losses,axis=0)[1]][0]
	
	w = lasso(train[0],train[1],opt_lambda)
	p = percent_match_supports(w,L.theta)

	print "\nsupports of w and theta match {}%".format(100*p)

if __name__=='__main__':

	main()	
	
	if False:

		X = numpy.loadtxt("X.txt")
		y = numpy.loadtxt("y.txt")
		(N,D) = X.shape
		w  = numpy.random.rand(D,1)

		for i in range(-5,6):
			Lambda 	= 10**i
			w_opt 	= minimize(ridge(Lambda), w)
			print Lambda, compute_loss(Lambda, w_opt.x)
