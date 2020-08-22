import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from numpy import genfromtxt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

#Normalize data
def normalize():
	global x
	mean_x=0
	var_x=0.0
	for i in range(x.size):
		mean_x = mean_x+(x.ix[i,0])
	mean_x = mean_x/(x.size)
	for i in range(x.size):
    		var_x = var_x+((x.ix[i,0])-mean_x)**2
	var_x = math.sqrt(var_x/(x.size))
	for i in range(x.size):
    		x.ix[i,0] = ((x.ix[i,0])-mean_x)/var_x

#Compute cost function
def J(theta1,theta0):
	summ0=0
	for i in range(x.size):
    		summ0 = summ0+(-(y.ix[i,0])+(theta1*(x.ix[i,0])+theta0))**2
	return (summ0/(2*x.size))
	
#Plot mesh of data values
def mesh_data():
	X0, Y0 = np.mgrid[0:2:50j,-1:1:50j]
	mesh = np.c_[X0.ravel(),Y0.ravel()]
	return X0,Y0,np.array([J(pt[1],pt[0]) for pt in mesh]).reshape(X0.shape)	


#Linear Regression
def part_a(alpha=0.001):
	res = np.array([])
	theta0 = 0
	theta1 = 0
	t = 0
	x0=1
	epsilon = 10**(-9)
	summ0=0
	for i in range(x.size):
    		summ0 = summ0+(-(y.ix[i,0])+(theta1*(x.ix[i,0])+theta0))**2
	j_theta = summ0/(2*x.size)

	while True:
		#update theta0
    		summ0=0
		for i in range(x.size):
			summ0 = summ0+(-(y.ix[i,0])+(theta1*(x.ix[i,0])+theta0))*x0
		prev_th0 = theta0
		theta0 = theta0-alpha*(summ0/x.size)
		#update theta1
		summ1=0
		for i in range(x.size):
			summ1 = summ1+(-(y.ix[i,0])+(theta1*(x.ix[i,0])+theta0))*(x.ix[i,0])
		prev_th1 = theta1
		theta1 = theta1-alpha*(summ1/x.size)
		t = t+1
		#update J(theta)
		summ0=0
		for i in range(x.size):
			summ0 = summ0+(-(y.ix[i,0])+(theta1*(x.ix[i,0])+theta0))**2
		prev_jth = j_theta
		j_theta = summ0/(2*x.size)
		res = np.append(res,[theta0,theta1,j_theta])
		#termination condition
		if t>7000:
			print 'Force stop'
			break
		if abs(j_theta - prev_jth) <=epsilon:
			break
		
	print "At eta = ",alpha
	print "Convergence condition : (J_new - J_prev < epsilon)"
	print "Parameters : Theta0 = ",theta0," Theta1 = ",theta1
	print "Error = ",j_theta 
	print "Converged after ",t, " iterations with epsilon = ",epsilon
	return theta0,theta1,res.reshape(t,3)

#Plot separator
def part_b(theta1,theta0):
	hyp = theta1*x+theta0
	data, = plt.plot(x,y,'gv',label = "Features") 
	hypoth, = plt.plot(x,hyp,'r',label = "Separator")
	plt.xlabel("Acidity of wine")
	plt.ylabel("Density of wine") 	
	plt.legend(handles=[data,hypoth])
	plt.title("Linear Regression")
	plt.savefig('q1_b.png')	
	plt.show()


#Plot 3d mesh of j_theta with x,y
def part_c(res,gap=0.2,flag=False):
	X0,Y0,j_val = mesh_data()
	fig = plt.figure()
    	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X0, Y0, j_val,cmap=cm.coolwarm)
    	plt.title("3D Mesh of J_theta with x,y")
	if flag is False:
		for it in res:
			T0 = [it[0]]
			T1 = [it[1]]
			T2 = [it[2]]
			ax.scatter(T0,T1,T2,color='k',linestyle='-',linewidth=3)
			plt.pause(gap)
		
	ax.set_zlabel('J(theta)')
    	ax.set_ylabel('theta_1')
    	ax.set_xlabel('theta_0')
		
	plt.plot(res[:,0],res[:,1],res[:,2],color='k',marker='o',markersize=2.5)	
	plt.savefig('q1_c.png')
	plt.show()
 
#Plot countours of error function
def part_d(res,gap=0.2,eta=0.001):
	X0,Y0,j_val = mesh_data() 
	fig = plt.figure()
    	ax = fig.add_subplot(111)
	plt.contour(X0, Y0, j_val,30)
    	ax.set_xlabel('theta_0')
    	ax.set_ylabel('theta_1')
  	for it in res:
		ax.scatter([it[0]],[it[1]],color='r',linestyle='-')
		plt.pause(gap)
	#plt.plot(res[:,0],res[:,1],color='r',marker='o',markersize=2.5)	
	plt.title("Contours at (eta=" + str(eta) + ")")
	plt.savefig('q1_d'+str(eta)+'.png')
	plt.show()

#Plot contours with different values of eta
def part_e():
	eta = [0.1,0.5,0.9,1.3,1.7,2.1,2.5]
	for i in eta:
		theta1,theta0,res=part_a(i)
		part_d(res,gap,i)
	

if(len(sys.argv)<5):
	print('Insufficient arguments')
	sys.exit()
x = pandas.read_csv(sys.argv[1],header=None)
y = pandas.read_csv(sys.argv[2],header=None)
normalize()
theta0,theta1,res=part_a(float(sys.argv[3]))
gap = float(sys.argv[4])
part_b(theta1,theta0)
part_c(res,gap)
part_c(res,gap,True)
part_d(res,gap,float(sys.argv[3]))
part_e()











	
