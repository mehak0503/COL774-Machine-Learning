import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from numpy import genfromtxt
from numpy import linalg as la
import matplotlib.lines as mline
import matplotlib.patches as mpatch

#Change y from [Alaska,Canada] to Class [0,1]
def mark_class():
	global y
	for i in range(m):
		if y_str[i]=='Alaska':
			y[i] = 0
		else:
			y[i] = 1
			

#Normalization of data
def normalize():
	global x
	for i in range(n):
		mean = np.mean(x[:,i])
		std = np.std(x[:,i])
		x[:,i] = (x[:,i]-mean)/std		


#Phi,mean0,mean1
def mean_params():
	elem_cls1 = np.sum(y)
	elem_cls0 = m-elem_cls1
	phi = elem_cls1/m
	x_alaska = np.array([])
	x_canada = np.array([])
	for i in range(m):
		if y[i]==0:
			x_alaska = np.append(x_alaska,x[i])
		else:
			x_canada = np.append(x_canada,x[i])
	x_alaska = x_alaska.reshape(x_alaska.size/2,2)
	x_canada = x_canada.reshape(x_canada.size/2,2)
	mean0 = np.mean(x_alaska,axis=0)
	mean1 = np.mean(x_canada,axis=0)
	x_als_mu = x_alaska - mean0
	x_cnd_mu = x_canada - mean1
	print 'phi',phi
	print 'mean',mean0,mean1
	return phi,mean0,mean1,x_alaska,x_canada

#Compute covariance matrix
def covar(flag=False):
	x_als_mu = x_alaska - mean0
	x_cnd_mu = x_canada - mean1
	co_var0 = np.matmul(np.transpose(x_als_mu),x_als_mu)/(m-np.sum(y))
	co_var1 = np.matmul(np.transpose(x_cnd_mu),x_cnd_mu)/np.sum(y)
	if flag is False:
		print 'covariance0'
		print co_var0
		print 'covariance1'
		print co_var1
		return co_var0,co_var1
	else:
		co_var = (co_var0+co_var1)/2
		print 'covariance'
		print co_var
		return co_var,co_var


#Closed Form Equations	
def part_a(flag=False):
	phi,mean0,mean1,x_alaska,x_canada = mean_params()
	covar(False)
	
	
#Plot training data
def part_b():
	clr = ['g' if i else 'c' for i in y]
	maping = {'g':'v','c':'*'}
	for i in range(m):
		plt.plot(x[i,0],x[i,1],color=clr[i],marker=maping[clr[i]]) 
	plt.xlabel("Feature_1")
	plt.ylabel("Feature_2")
	clas0 = mpatch.Patch(color='cyan', label='Class 0 (Alaska)')
	clas1 = mpatch.Patch(color='green', label='Class 1 (Canada)') 	
	plt.legend(handles=[clas0,clas1])
	plt.title("Feature points")
	plt.savefig('q4_b.png')	
	return plt
	

#Decision boundary for same covariance
def part_c(phi,mean0,mean1,co_var):
	sig_inv = la.inv(co_var)
	mean0 = mean0.reshape(2,1)
	mean1 = mean1.reshape(2,1)
	mu0_tr = np.transpose(mean0)
	mu1_tr = np.transpose(mean1)
	alpha = 2*(np.matmul(np.transpose(mean0-mean1),sig_inv))
	beta = np.matmul(np.matmul(mu0_tr,sig_inv),mean0)-np.matmul(np.matmul(mu1_tr,sig_inv),mean1)-(2*np.log(1/phi-1))
	#Equation of line : alpha*X = beta	
	x1 = (beta-alpha[0,0]*(x[:,0]).reshape(m,1))/alpha[0,1]
	plt = part_b()
	plt.title("Linear Boundary")
	plt.plot(x[:,0],x1,'k--',lw=0.3)
	plt.savefig('q4_c.png')	
	return plt	
	
#Closed Form Equations for different covariance matrix
def part_d():
	return part_a(True)


#Quadratic boundary for different covariance
def part_e():
	global mean0,mean1
	sig0_inv = la.inv(covar0)
	sig1_inv = la.inv(covar1)
	mean0 = mean0.reshape(2,1)
	mean1 = mean1.reshape(2,1)
	mu0_tr = np.transpose(mean0)
	mu1_tr = np.transpose(mean1)
	a = sig0_inv-sig1_inv
	b = -2*(np.matmul(mu0_tr,sig0_inv)-np.matmul(mu1_tr,sig1_inv))
	c = np.matmul(mu0_tr,np.matmul(sig0_inv,mean0))-np.matmul(mu1_tr,np.matmul(sig1_inv,mean1))-2*(np.log((1/phi)-1)*(la.det(covar1))/(la.det(covar0)))
	x0,x1 = np.mgrid[-10:10:100j,-10:10:100j]	
	fin = np.c_[x0.ravel(),x1.ravel()]
	bndry = np.array([eq_val(a,b,c,pt) for pt in fin])
	bndry = bndry.reshape(x0.shape)
	plt.contour(x0,x1,bndry,[0],colors='r')	
	plt.title('Linear vs quadratic boundary')
	plt.savefig('q4_e.png')
	plt.show()

#Equation value
def eq_val(a,b,c,x):
        	return np.matmul(np.transpose(x),np.matmul(a,x)) + np.matmul(b,x) + float(c)
		

if(len(sys.argv)<4):
	print('Insufficient arguments')
	sys.exit()

x = genfromtxt(str(sys.argv[1]))
y_str = genfromtxt(str(sys.argv[2]),dtype='str')
m = np.size(x,0)
if x.ndim > 1:
	n = np.size(x,1)
else:
	n = 1
	x = x.reshape(m,n)
y_str = y_str.reshape(m,1)
y = np.zeros((m,1))
mark_class()
normalize()
if sys.argv[3]=='0':
	phi,mean0,mean1,x_alaska,x_canada= mean_params()
	covar0,covar1 = covar(True)
	plt = part_b()
	plt.show()
	plt = part_c(phi,mean0,mean1,covar0)
	plt.show()
else:
	phi,mean0,mean1,x_alaska,x_canada= mean_params()
	covar0,covar1 = covar(True)
	plt = part_c(phi,mean0,mean1,covar0)
	covar0,covar1=covar(False)	
	part_e()
