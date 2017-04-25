import numpy

from scipy.optimize import minimize
from hw1_sgd.hw1_skeleton_code import regularized_grad_descent, regularized_batch_gradient_descent_plotter, feature_normalization

def ridge(Lambda):
	def ridge_obj(theta):
		return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N) + Lambda*(numpy.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(Lambda, theta):
	return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N)

class LinearSystem(object):
	"""data matrix of dimensions mxd (num_examples x num_dimensions)"""
	def __init__(self,m=150,d=75,p=.5):
		self.m = m
		self.d = d
		self.p = p

		self.design_matrix()
		self.theta()
		self.y()

	def design_matrix(self):
		self.X = numpy.random.rand(self.m,self.d)

	# theta is a sparse vector; theta[i] != 0 w probability self.p
	def theta(self,fixed_theta=True):
		self.theta = numpy.zeros(self.d)
		if fixed_theta:
			for i in xrange(10):
				if i%2==0:
					self.theta[i] = 10.
				else:
					self.theta[i] = -10.
		else:
			for i in xrange(10):
				if numpy.random.rand() < self.p:
					self.theta[i] = 10.
				else:
					self.theta[i] = -10.

	# generates outputs y = x*theta + e, e ~ N(0,1)
	def y(self):
		self.y = self.X.dot(self.theta) + numpy.random.randn(self.m)

	def split(self):
		train_set = (self.X[:80], self.y[:80])
		valid_set = (self.X[80:101],self.y[80:101])
		test_set  = (self.X[101:],self.y[101:])
		return train_set, valid_set, test_set

def lasso(X,y,lamb,eps=10e-6):
	n = y.shape()[0]
	diff = np.inf
	while diff > eps:
		for i in xrange(n):
			c = x[]

if __name__=='__main__':
	L 						= LinearSystem()
	train, valid, test 		= L.split()
	lambdas  				= (.000001, .000005 , .00001 , .00005,  .0001, .0005, .001)
	theta_hist, loss_hist 	= regularized_grad_descent(train[0],train[1])
	for i in xrange(L.d):
		print theta_hist[-1][i], L.theta[i]
	print loss_hist[-1]
	
	# lambdas                 = regularized_batch_gradient_descent_plotter(train[0],train[1],valid[0],valid[1],lambdas=lambdas)

	if False:
		X 			= numpy.loadtxt("X.txt")
		y 			= numpy.loadtxt("y.txt")
		(N,D) 		= X.shape
		w 			= numpy.random.rand(D,1)

		for i in range(-5,6):
			Lambda 	= 10**i
			w_opt 	= minimize(ridge(Lambda), w)
			print Lambda, compute_loss(Lambda, w_opt.x)
