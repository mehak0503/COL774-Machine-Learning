import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from numpy import genfromtxt
from numpy import linalg as la

#Compute h
def h(theta0,theta1):
	return float(theta1)*x+theta0

#Normalization of data
def normalize():
	mean = np.mean(x)
	std = np.std(x)
	for i in range(m):
		x[i] = (x[i] - mean)/std

#Plot of locally weighted
def plot_data(theta0,theta1):
	val = h(theta0,theta1)
	pp, = plt.plot(x[:,1],y,'gv',label = "Features") 
	pq, = plt.plot(x[:,1],val[:,1],'k',label = "Model")
	plt.xlabel("X")
	plt.ylabel("Y") 	
	plt.legend(handles=[pp,pq])
	plt.title("Unweighted linear regression")
	plt.savefig('q2_a.png')	
	plt.show()

#Unweighted Linear Regression
def part_a():
	theta = np.zeros((n+1,1))
	tr_x = np.transpose(x)
	dmy = np.matmul(tr_x,x)
	dmy = la.inv(dmy)
	dmy = np.matmul(dmy,tr_x)
	theta = np.matmul(dmy,y)
	plot_data(theta[0,0],theta[1,0])

#Compute Weight matrix
def wgt_matrix(x_pt,tau):
	return np.diag(np.exp(-((x_pt-x[:,1])**2)*0.5/(tau*tau)))


#Plot 
def plot_ret():
	plt.ylabel("Y")
	plt.xlabel("Feature") 			
	return plt


#Weighted Linear Regression
def part_b(tau):
	qry_x = np.linspace(np.amin(x[:,1]), np.amax(x[:,1]))
	res = np.array([])
	for i in qry_x:
		w = wgt_matrix(i,tau)
		theta = np.matmul(la.inv(np.matmul(np.transpose(x),np.matmul(w,x))),np.matmul(np.transpose(x),np.matmul(w,y)))
		res = np.append(res,np.matmul(np.transpose(theta),np.array([[1],[i]])))
	plt = plot_ret()
	title = "LWR for tau " + str(tau)	
	hyp, = plt.plot(qry_x,res,'k',label = title)	
	data, = plt.plot(x[:,1],y,'cv',label="Features")
	plt.legend(handles=[data,hyp])	
	plt.title(title)	
	plt.savefig('q2_b'+str(tau)+'.png')
	plt.show()

def part_c():
	tau = [0.1,0.3,2,10]
	for i in tau:
		part_b(i)

if(len(sys.argv)<4):
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
part_a()
part_b(float(sys.argv[3]))
part_c()
