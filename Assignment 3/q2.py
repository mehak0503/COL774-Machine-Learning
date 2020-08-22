import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from neural_net import NeuralNet
from sklearn.metrics import confusion_matrix
from confmat import plot_confusion
import pandas as pd
import timeit
from matplotlib import pyplot as plt
import sys
f_ptr = open('m.txt','a')

def lab_enc(d):
	le = LabelEncoder()
	int_e = le.fit_transform(d)
	int_e = int_e.reshape(d.size,1)
	return int_e

def hot_enc(d):
	oe = OneHotEncoder(sparse=False)
	encd = oe.fit_transform(d)
	return encd

def plot_metric(tt,l,s):
	plt.plot(l,tt[:,0])
	plt.plot(l,tt[:,1])
	plt.xlabel("No of hidden layer units")
	plt.ylabel("Accuracy")
	plt.legend(["Train "+str(round(max(tt[:,0]),1)),"Test "+str(round(max(tt[:,1]),1))])
	plt.savefig(s+".png")
	#plt.show()
	plt.close()

def enc_x():
	global x,m,n
	y = x[:,-1]
	x = np.delete(x,-1,axis=1)
	for i in range(n-1):
		int_encd = lab_enc(x[:,i])
		hot_encd = hot_enc(int_encd)
		x = np.append(x,hot_encd,axis=1)
		
	for i in range(n-1):
		x = np.delete(x,0,axis=1)
	x = np.append(x,y.reshape(y.size,1),axis=1)	
	m,n = x.shape
	

def enc_t():
	global tests,mt,nt
	y = tests[:,-1]
	tests = np.delete(tests,-1,axis=1)
	for i in range(nt-1):
		int_encd = lab_enc(tests[:,i])
		hot_encd = hot_enc(int_encd)
		tests = np.append(tests,hot_encd,axis=1)
		
	for i in range(nt-1):
		tests = np.delete(tests,0,axis=1)
	tests = np.append(tests,y.reshape(y.size,1),axis=1)	
	mt,nt = tests.shape

def part_a():
	enc_x()
	enc_t()
	
def part_b(file_n):
	f = open(file_n,"r")
	ip = int(f.readline())
	op = int(f.readline())
	batch = int(f.readline())
	n = int(f.readline())
	h = (f.readline()).rstrip().split(" ")
	h = map(int,h)
	h = [ip]+h+[op]
	if f.readline()=="relu\n":
		non_lin = 1
	else:
		non_lin = 0
	if f.readline()=="fixed\n":
		eta = 0
	else:
		eta = 1
	print ip,op,batch,n
	print h
	print non_lin,eta
	start = timeit.default_timer()
	net = NeuralNet(h,bool(non_lin))
	net.grad_des(x[:,0:-1],x[:,-1],batch,bool(eta))
	stop = timeit.default_timer()
	t_acc = 100*net.score(x[:,0:-1],x[:,-1])
	ts_acc = 100*net.score(tests[:,0:-1],tests[:,-1])
	print "Train accuracy ",t_acc
	print "Test accuracy ",ts_acc
	print "Training time ",(stop-start)
	conf = confusion_matrix(tests[:,-1].tolist(),net.pred(tests[:,0:-1]))
	plot_confusion(conf,list(set(tests[:,-1].flatten().tolist())),"For layers "+str(h))
	

def part_c(eta_a=False,rlu=False):
	tt = np.zeros((5,2))
	m = 0
	h = [85,0,10]
	l = [5,10,15,20,25]
	for i in [5,10,15,20,25]:
		print "For 1 layer ",i,eta_a,rlu
		h[1] = i
		start = timeit.default_timer()
		net = NeuralNet(h,rlu)
		net.grad_des(x[:,0:-1],x[:,-1],100,eta_a)
		stop = timeit.default_timer()
		t_acc = 100*net.score(x[:,0:-1],x[:,-1])
		ts_acc = 100*net.score(tests[:,0:-1],tests[:,-1])		
		f_ptr.write("\nFor single layer "+str(eta_a)+str(rlu))
		f_ptr.write(str(i))
		f_ptr.write("\nTraining acc ")
		f_ptr.write(str(t_acc))
		f_ptr.write("\nTesting acc ")
		f_ptr.write(str(ts_acc))
		f_ptr.write("\nTrainig time ")
		f_ptr.write(str(stop-start))
		print "Train accuracy ",t_acc
		print "Test accuracy ",ts_acc
		print "Training time ",(stop-start)
		tt[m,0] = t_acc
		tt[m,1] = ts_acc
		m = m+1	
		conf = confusion_matrix(tests[:,-1].tolist(),net.pred(tests[:,0:-1]))
		plot_confusion(conf,list(set(tests[:,-1].flatten().tolist())),"For layers "+str(h)+str(eta_a)+str(rlu))
	print tt
	plot_metric(tt,l,"For one hidden layer "+str(eta_a)+str(rlu))
	

def part_d(eta_a=False,rlu=False):
	tt = np.zeros((5,2))
	m = 0
	h = [85,0,0,10]
	l = [5,10,15,20,25]
	for i in [5,10,15,20,25]:
		print "For 2 layer ",i,eta_a,rlu
		h[1] = i
		h[2] = i
		start = timeit.default_timer()
		net = NeuralNet(h,rlu)
		net.grad_des(x[:,0:-1],x[:,-1],100,eta_a)
		stop = timeit.default_timer()
		t_acc = 100*net.score(x[:,0:-1],x[:,-1])
		ts_acc = 100*net.score(tests[:,0:-1],tests[:,-1])
		f_ptr.write("\nFor double layer "+str(eta_a)+str(rlu))
		f_ptr.write(str(i))
		f_ptr.write("\nTraining acc ")
		f_ptr.write(str(t_acc))
		f_ptr.write("\nTesting acc ")
		f_ptr.write(str(ts_acc))
		f_ptr.write("\nTrainig time ")
		f_ptr.write(str(stop-start))
		print "Train accuracy ",t_acc
		print "Test accuracy ",ts_acc
		print "Training time ",(stop-start)
		tt[m,0] = t_acc
		tt[m,1] = ts_acc
		m = m+1	
		conf = confusion_matrix(tests[:,-1].tolist(),net.pred(tests[:,0:-1]))
		plot_confusion(conf,list(set(tests[:,-1].flatten().tolist())),"For 2 layers "+str(h)+str(eta_a)+str(rlu))
	print tt
	plot_metric(tt,l,"For two hidden layers "+str(eta_a)+str(rlu))

d = pd.read_csv(sys.argv[2],header=None)
x = d.iloc[:,:].values 
#x = genfromtxt(sys.argv[2],delimiter=',',dtype=int)
#x = genfromtxt('poker-hand-training.data.csv',delimiter=',',dtype=int)
m,n = x.shape

t = pd.read_csv(sys.argv[3],header=None)
tests = t.iloc[:,:].values
#tests = genfromtxt(sys.argv[3],delimiter=',',dtype=int)
#tests = genfromtxt('poker-hand-testing.data.csv',delimiter=',',dtype=int)
mt,nt = tests.shape

#part_a()
part_b(sys.argv[1])
#part_c()
#part_d()
#f_ptr.write("\n\n======================\nWith adaptive learning rate \n")
#part_c(True)
#part_d(True)
#f_ptr.write("\n\n======================\nWith adaptive learning rate and relu \n")
#part_c(True,True)
#part_d(True,True)


f_ptr.close()
