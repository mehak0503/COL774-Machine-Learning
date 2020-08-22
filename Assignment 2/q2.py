import numpy as np
from numpy import genfromtxt
import cvxopt
from collections import Counter
import timeit
import matplotlib.pyplot as plt
from numpy import linalg 
from sklearn.metrics import confusion_matrix
import pickle
from svmutil import svm_parameter,svm_predict,svm_train,svm_problem
from random import randint
import confmat
import sys
x = []
y = []
test_x = []
test_y = []
v_set_x = np.array([])
v_set_y = np.array([])
acc = 0
acc_v = 0

def g(z):
	return 1/(1+np.exp(-z))

def compute_b(w,alpha):
	b = 0
	for i in range(m):
		w_t_x = 0
		for j in range(len(alpha)):
			w_t_x = w_t_x + alpha[j,0]*y[j]*g_kernel(x[j,:],x[i,:])
		b = b+(y[i]-w_t_x)

	return b/float(len(alpha))

def linear_kernel():
	K = np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			K[i,j] = y[i,0]*y[j,0]*l_kernel(x[i,:],x[j,:])
	return K
	
def gaussian_kernel():
	K = np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			K[i,j] = y[i,0]*y[j,0]*g_kernel(x[i,:],x[j,:])			
	return K
					

def l_kernel(x1,x2):
	return np.dot(np.transpose(x1),x2)
	
def g_kernel(x1,x2,gamma = 0.005):
	return np.exp(-gamma*(linalg.norm(x1-x2)**2))

def comp_params(flag):
	if flag==True:
		Q = cvxopt.matrix(gaussian_kernel()*(1.0))
	else:
		Q = cvxopt.matrix(linear_kernel()*(1.0))
	b = cvxopt.matrix(np.ones((m,1))*(-1))
	c = cvxopt.matrix(0.0)
	A = cvxopt.matrix(y, (1,m))
	return Q,b,c,A

def param_st(C=1.0):
	tmp1 = np.diag(np.ones(m) * -1)
        tmp2 = np.identity(m)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(m)
        tmp2 = np.ones(m) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
	return G,h

def SVM_alpha(flag):
	Q,b,c,A = comp_params(flag)
	G,h = param_st()
	sol= cvxopt.solvers.qp(Q, b, G, h, A, c)
        alpha = np.ravel(sol['x'])
        sv = alpha > 1e-5
	svv = alpha <= 1.0
        ind = np.arange(len(alpha))[sv]
	indd = np.arange(len(alpha))[svv]
	ind = [i for i in indd if i in ind]
        alpha_sv = alpha[sv]
	alpha_svv = alpha[svv]
	alpha_sv = [i for i in alpha_svv if i in alpha_sv]
        print("%d support vectors out of %d points after 1e-5" % (len(alpha_sv), m))
	#print "Support vectors are ",ind
	#print "Alpha values for support vectors are ",alpha_sv 	
	#return alpha_sv,ind
	return alpha,ind

def SVM(flag,n=3.0,p=4.0):
	alpha,ind = SVM_alpha(flag)	
	alpha = np.reshape(alpha,(len(alpha),1))
	if flag==False:
		w = np.sum(y*alpha*x,axis=0)	
		print "Weight vector w is  ",w
	else:
		w = 0
	b = compute_b(w,alpha)
	print "Intercept term b is ",b
	print "For n = ",n," p = ",p
	return alpha,w,b

def SVM_test(s,flag=False):
	start = timeit.default_timer()
	alpha,w,b = SVM(flag)
	stop = timeit.default_timer()
	print "Training time with "+s+"kernel",stop-start,"seconds"
	pred_y = np.zeros((mm,1))
	if flag==True:
		for i in range(mm):
			for j in range(m):
				pred_y[i,0] = pred_y[i,0]+alpha[j,0]*y[j]*g_kernel(x[j,:],test_x[i,:])

		pred_y = pred_y+b
	else:
		print w.shape
		print test_x.shape
		pred_y = np.matmul(test_x,np.transpose(w))+b
	pred = [1]*mm
	a = [i for i in range(mm) if g(pred_y[i])<0.5]
	for i in a:
		pred[i] = -1
	acc = 0
	for i in range(mm):
		if pred[i]==test_y[i,0]:
			acc = acc+1
	print "Accuracy on test set with "+s+" kernel is ",float(acc)*100/float(mm)
def part_a():
	print "SVM with Linear Kernel (CVXOPT)\n"
	SVM_test('linear')

def part_b():
	print "\n\nSVM with Gaussian Kernel (CVXOPT)\n"
	SVM_test('gaussian',True)
	print "\n\n"

def libsvm_param(p):
	prob = svm_problem(y,x.tolist())
	param = svm_parameter(p)
	return prob,param

def libsvm(prob,param,par_s,s,flag=False,flag_x=False,flag_v=False,conf=False):
	global acc,acc_v	
	param.parse_options(par_s)
	print "c ",param.C
	start = timeit.default_timer()
	model = svm_train(prob,param)
	stop = timeit.default_timer()
	print "Training time with libsvm "+s+" kernel",stop-start,"seconds"
	l = model.l
	print "Libsvm weight vector w is "
	alph = np.zeros((m,1))
	if flag==False:	
		for i in range(l):
			ind = model.sv_indices[i]
			alph[ind,0] = model.sv_coef[0][i]
		w = np.matmul(np.transpose(alph*y),x)
		print "Weight vector w is ",w
	print "Intercept form b is ",b
	if flag_v==True:
		_, acc_v, _ = svm_predict(v_set_y, v_set_x.tolist(), model)
		print("Accuracy over validation test set using libsvm with "+s+" kernel: ", acc_v)
	if flag_x==True:		
		_, acc_x, _ = svm_predict(y, x.tolist(), model)
		print("Accuracy over training set using libsvm with "+s+" kernel: ", acc_x)
	pred_val, acc, _ = svm_predict(test_y, test_x.tolist(), model)
	print("Accuracy over test set using libsvm with "+s+" kernel: ", acc)
	if conf==True:
		conf = confusion_matrix(test_y.tolist(),pred_val)
		confmat.plot_confusion(conf,list(set(test_y.flatten().tolist())),"SVM (Multi-Classification)")
	

	
def part_c():
	prob,param = libsvm_param("-q -s 0 -c 1")
	libsvm(prob,param,"-t 0","linear")
	libsvm(prob,param,"-t 2 -g 0.05","gaussian",True)

def read_train(neg,p):
	global x,y,m,n
	m,n = f.shape
	x = [f[i,:] for i in range(m) if f[i,784]==neg or f[i,784]==p]
	x = np.array(x)
	y = x[:,784]
	y = np.array(y)
	x = x/255.0
	x = np.delete(x, np.s_[-1:], axis=1)
	m,n = x.shape
	x = np.reshape(x,(m,784))
	y = np.reshape(y,(m,1))
	for i in range(m):
		if y[i,0]==neg:
			y[i,0]=-1
		else:
			y[i,0]=1	
	
def read_test(neg,p):
	global mm,nn,test_x,test_y
	test_x = [f_test[i,:] for i in range(mm) if f_test[i,784]==neg or f_test[i,784]==p]
	test_x = np.array(test_x)
	test_y = test_x[:,784]
	test_x = test_x/255.0
	test_x = np.delete(test_x, np.s_[-1:], axis=1)
	mm,nn = test_x.shape
	test_x = np.reshape(test_x,(mm,784))
	test_y = np.reshape(test_y,(mm,1))
	for i in range(mm):
		if test_y[i,0]==neg:
			test_y[i,0]=-1
		else:
			test_y[i,0]=1

def classifiers(flag=False):
	global x,y,m,n
	gauss_classifiers = []	
	for k in range(10):
		iter = [k+i for i in range(10-k)] 
		for l in iter:
			if k==l:
				continue
			read_train(k,l)
			alpha,w,b = SVM(True,k,l)
			gauss_classifiers.append([k,l,alpha,b])
	with open("gauss_classifiers.pkl","wb") as pickle_out:
                pickle.dump(gauss_classifiers,pickle_out)
	return gauss_classifiers
				
def read():
	global m,n,x,y
	x = f[:,:]
	x = np.array(x)
	y = x[:,784]
	y = np.array(y)
	x = x/255.0
	x = np.delete(x, np.s_[-1:], axis=1)
	m,n = x.shape
	x = np.reshape(x,(m,784))
	y = np.reshape(y,(m,1))	
	

def read_t():
	global mm,nn,test_x,test_y
	test_x = f_test[:,:]	
	test_x = np.array(test_x)
	test_y = test_x[:,784]
	test_x = test_x/255.0
	test_x = np.delete(test_x, np.s_[-1:], axis=1)
	mm,nn = test_x.shape
	test_x = np.reshape(test_x,(mm,784))
	test_y = np.reshape(test_y,(mm,1))
	
	

def test(gauss_classifiers,xx):
	pred_gau = []
	for itx in xx:
		p_gau = []
		pred_y = 0
		for it in gauss_classifiers:
			n,p,alpha,b = it
			for j in range(len(alpha)):
				pred_y=pred_y+alpha[j,0]*y[j]*g_kernel(x[j,:],itx)

			pred_y = pred_y+b
			if pred_y>0:
				p_gau.append(p)
			elif pred_y==0:
				p_gau.append(max(p,n))
			else:
				p_gau.append(n)	
			
		pred_gau.append(max(p_gau,key=p_gau.count))
		
	return pred_gau

def part_d():
	#with open("gauss_classifiers.pkl","rb") as pickle_in:
	#	gauss_classifiers = pickle.load(pickle_in)
	gauss_classifiers = classifiers()
	read()
	pred_gau = test(gauss_classifiers,x)
	pred_gau_t = test(gauss_classifiers,test_x)
	acc_gau = 0
	acc_gau_t = 0
	for i in range(m):
		if pred_gau[i]==y[i]:
			acc_gau = acc_gau+1
		print "For ",i," actual ",y[i]," gauss ",pred_gau[i]
	for i in range(mm):
		if pred_gau_t[i]==test_y[i]:
			acc_gau_t = acc_gau_t+1
		print "For ",i," actual ",y[i]," gauss ",pred_gau_t[i]
	print "Accuracy for multi-class SVM with gaussian kernel over training set ",float(acc_gau)/float(m)
	print "Accuracy for multi-class SVM with gaussian kernel over test set ",float(acc_gau_t)/float(mm)



def part_e():
	prob,param = libsvm_param("-q -s 0 -c 1")
	libsvm(prob,param,"-t 2 -g 0.05","gaussian",True,True)


def part_f():
	prob,param = libsvm_param("-q -s 0 -c 1")
	libsvm(prob,param,"-t 2 -g 0.05","gaussian",True,False,False,True)

def part_g():
	global x,m,n,v_set_x,v_set_y,y
	v = int(0.10*m)
	v_set_x = np.zeros((v,784))
	v_set_y = np.zeros((v,1))
	res = np.zeros((5,3))
	for i in range(v):
		m,n = x.shape
		r = randint(0,m-1)
		v_set_x[i,:] = x[r,:]
		v_set_y[i,:] = y[r]
		x = np.delete(x,r,0)
		y = np.delete(y,r,0)
	m,n = x.shape
	i = 0
	for c in [10**(-5),10**(-3),1,5,10]:
		print "For C = ",c
		s = " -t 2 -g 0.05 -c %f" % c
		prob,param = libsvm_param("-q -s 0")
		libsvm(prob,param,s,"gaussian",True,False,True)
		res[i,0]=c
		res[i,1]=acc_v[0]
		res[i,2]=acc[0]
		i=i+1
	print res
	plot_res(res)

def plot_res(res):
	res = np.array(res)
	res = np.reshape(res,(5,3))
	pp, = plt.plot(res[:,0],res[:,1],'-rx',label = "Validation Set Accuracy") 
	pq, = plt.plot(res[:,0],res[:,2],'-bo',label = "Test Set Accuracy")
	plt.xlabel("C-Values")
	plt.ylabel("Accuracy") 	
	plt.xscale('log')
	plt.legend(handles=[pp,pq])
	plt.title("SVM C-Value estimation")
	plt.savefig('svm_c.png')	
	plt.show()

	
if(len(sys.argv)<5):
	print('Insufficient arguments')
	sys.exit()
	
f = genfromtxt(sys.argv[1],delimiter=',')
m,n = f.shape
f_test = genfromtxt(sys.argv[2],delimiter=',')
mm,nn = f_test.shape

if int(sys.argv[3])== 0:
	read_train(3,4)
	read_test(3,4)
	if sys.argv[4]=='a':
		part_a()
	elif sys.argv[4]=='b':
		part_b()
	elif sys.argv[4]=='c':
		part_c()
	else:
		print "Incorrect part number"

else:
	read()
	read_t()
	if sys.argv[4]=='a':
		part_d()
	elif sys.argv[4]=='b':
		part_e()
	elif sys.argv[4]=='c':
		part_f()
	elif sys.argv[4]=='d':
		part_g()
	else:
		print "Incorrect part number"
	




