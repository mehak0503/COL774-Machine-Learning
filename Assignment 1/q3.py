import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import genfromtxt
import matplotlib.patches as mpatch


#Normalization of data
def normalize():
	global x
	for i in range(n):
		mean = np.mean(x[:,i])
		std = np.std(x[:,i])
		x[:,i] = (x[:,i]-mean)/std		

#Sigmoid function
def g(z):
	return 1/(1+np.exp(-z))

#Hypothesis function
def h(theta,x):
	return g(np.matmul(x,theta))

#Gradient function
def grad(theta):
	return np.matmul(np.transpose(x),(y-h(theta,x)))


#Log likehood function
def l(theta):
	h_th = h(theta,x)
	return -1*np.sum(y*np.log(h_th)+(1-y)*np.log(1-h_th))


#Hessian function
def hessian(theta):
	g_val = h(theta,x)
	return np.matmul(np.transpose(x),np.matmul(np.diagflat(g_val*(1-g_val)),x))

#Newton method
def part_a():
	iters = 0
	epsilon = 10**(-12)
	theta = np.zeros((n+1,1))	
	l_n = l(theta)
	converged = False
	while not converged:
		theta = theta + np.matmul(np.linalg.pinv(hessian(theta)),grad(theta))		
		l_p = l_n
		l_n = l(theta)
		iters = iters+1
		if abs(l_n-l_p)<epsilon:
			converged = True

	print "Parameters = "
	print theta
	print "Error = ",l_n
	print "Converged after ",iters," iterations"
	return theta

#Find x2 using h_theta = theta2 * x2 + theta1 * x1 + theta0 = 0
def x2(th):
	return -(th[0]+th[1]*x[:,1])/th[2] 

#Plot data
def plot_data():
	clr = ['g' if i else 'c' for i in y]
	maping = {'g':'v','c':'*'}
	for i in range(m):
		plt.plot(x[i,1],x[i,2],color=clr[i],marker=maping[clr[i]]) 
	#line, = plt.plot(x[:,1],x2(theta),'k',label = "Linear Separator")
	plt.xlabel("Feature_1 (x1)")
	plt.ylabel("Feature_2 (x2)")
	clas0 = mpatch.Patch(color='cyan', label='Class 0')
	clas1 = mpatch.Patch(color='green', label='Class 1') 	
	plt.legend(handles=[clas0,clas1])
	plt.title("Training data")
	return plt


#Plot of features and h(theta)
def part_b(theta):
	plt = plot_data()	
	plt.plot(x[:,1],x2(theta),'k',label = "Linear Separator")
	plt.title("Logistic Regression")
	plt.savefig('q3_b.png')	
	plt.show()

if(len(sys.argv)<3):
	print('Insufficient arguments')
	sys.exit()

x = genfromtxt(sys.argv[1],delimiter=',')
y = genfromtxt(sys.argv[2],delimiter=',')

m = np.size(x,0)
if x.ndim > 1:
	n = np.size(x,1)
else:
	n = 1
	x = x.reshape(m,n)
y = y.reshape(m,1)
normalize()
x0 = np.ones((m,1))
x = np.append(x0,x,axis=1)
theta = part_a()
plt = plot_data()
plt.show()
part_b(theta)
