import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys

def lab_enc(d):
	le = LabelEncoder()
	int_e = le.fit_transform(d)
	int_e = int_e.reshape(d.size,1)
	return int_e

def hot_enc(d):
	oe = OneHotEncoder(sparse=False)
	encd = oe.fit_transform(d)
	return encd


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
	print x.shape
	

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
	print tests.shape

def part_a(x_file,t_file):
	enc_x()
	enc_t()
	np.savetxt(x_file,x,delimiter=",")
	np.savetxt(t_file,tests,delimiter=",")


x = genfromtxt(sys.argv[1],delimiter=',',dtype=int)
#x = genfromtxt('poker-hand-training.data.csv',delimiter=',',dtype=int)
m,n = x.shape

tests = genfromtxt(sys.argv[2],delimiter=',',dtype=int)
#tests = genfromtxt('poker-hand-testing.data.csv',delimiter=',',dtype=int)
mt,nt = tests.shape

part_a(sys.argv[3],sys.argv[4])

