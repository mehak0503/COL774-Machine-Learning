import numpy as np
import pandas as pd
from PIL import Image
import os
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pickle
from itertools import combinations
from svmutil import svm_parameter,svm_predict,svm_train,svm_problem
from collections import Counter
import timeit
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
import random
import sys
from sklearn.utils import shuffle
import csv

def libsvm_param(p):
    prob = svm_problem(y.flatten().tolist(),x.tolist())
    param = svm_parameter(p)
    return prob,param

def libsvm(prob,param,par_s,s,flag=False,flag_x=False,flag_v=False,conf=False):
    global x,y	
    param.parse_options(par_s)
    print("------Start Training------")
    start = timeit.default_timer()
    model = svm_train(prob,param)
    stop = timeit.default_timer()
    print("Training time with libsvm "+s+" kernel",stop-start,"seconds")

    if flag_x==True:		
        pred_x, acc_x, _ = svm_predict(y.flatten().tolist(), x.tolist(), model)
        print("Accuracy over training set using libsvm with "+s+" kernel: ", acc_x)
        f_macro(y.flatten().tolist(),pred_x,s+" for training set ")
    
    del x
    del y
    
    if flag_v==True:
        df = pd.read_csv('valid_x.csv',header=None)
        v_set_x = df.iloc[:,:].values 
        df = pd.read_csv('rewards.csv',header=None)
        v_set_y = df.iloc[:,:].values 
        pred_v, acc_v, _ = svm_predict(v_set_y.flatten().tolist(), v_set_x.tolist(), model)
        print("Accuracy over validation test set using libsvm with "+s+" kernel: ", acc_v)
        f_macro(v_set_y.flatten().tolist(),pred_v,s+" for validation set ")
    del v_set_x
    del v_set_y
    df = pd.read_csv('test_x.csv',header=None)
    t_set_x = df.iloc[:,:].values
    m,n = t_set_x.shape
    t_set_y = [0]*m
    pred_t, acc_t, _ = svm_predict(t_set_y, t_set_x.tolist(), model)
    fin_pred = []
    for i in range(len(pred_t)):
        fin_pred.append([i,pred_t[i]])
    with open('test_pred.csv','wb') as file:
        writer = csv.writer(file)
        writer.writerow(["id","Prediction"])
        np.savetxt(file,np.array(fin_pred), delimiter=",",fmt="%i")

	
def lin():
    prob,param = libsvm_param("-q -s 0 -c 1")
    libsvm(prob,param,"-t 0","linear",False,True,True,True)

def gau():
    prob,param = libsvm_param("-q -s 0 -c 1")
    libsvm(prob,param,"-t 2 -g 0.05","gaussian",False,True,True,True)


def f_macro(y_orig,y_pred,s):
    f_score = f1_score(y_orig, y_pred, average=None)
    print("F1_score for "+str(s)+" is: ")
    print(f_score)
    
    f_mac = f1_score(y_orig, y_pred, average='macro')  
    print("F1_macro_score for "+str(s)+" is: ")
    print(f_mac)

    f_mic = f1_score(y_orig, y_pred, average='micro')  
    print("F1_micro_score for "+str(s)+" is: ")
    print(f_mic)

def cross_vald_lin():
    c_val = [10**(-2),10**(-1),1,5,10]
    prob,param = libsvm_param("-q -s 0")
    for c in c_val:
        print("\n\n\nFor C = ",c)
        s = " -t 0 -v 10 -c %f" %c
        libsvm(prob,param,s,"linear",True,False,True)
		

def cross_vald_gauss():
    c_val = [10**(-2),10**(-1),1,5,10]
    gamma = [0.01,0.05,0.1,0.3,0.5]
    prob,param = libsvm_param("-q -s 0")
    for c in c_val:
        for g in gamma:
            print("\n\n\nFor C = ",c)
            s = " -t 2 -v 10 -c %f" %c
            s = s+" -g %f" %g
            libsvm(prob,param,s,"gaussian",True,False,True)
		


df = pd.read_csv('train_x_0.csv',header=None)
x0 = df.iloc[:,:].values 
m,n = x0.shape

y0 = np.array([0]*m)
y0 = y0.reshape(-1,1)
#no_of_x0 = int(m*0.4)
#indx = random.sample(range(m),no_of_x0)
#indx = random.sample(range(m),18000)


#x0_new = [x0[i] for i in indx]
#y0_new = [y0[i] for i in indx]
#y0_new = [0]*len(indx)
#del x0

df = pd.read_csv('train_x_1.csv',header=None)
x1 = df.iloc[:,:].values 
m,n = x1.shape
y1 = np.array([1]*m)
y1 = y1.reshape(-1,1)
#no_of_x1 = int(m*0.1)
#indx = random.sample(range(m),no_of_x1)
#indx = random.sample(range(m),7000)

#x1_new = [x1[i] for i in indx]
#y1_new = [1]*len(indx)
#del x1

#x0_new = np.array(x0_new)
#y0_new = np.array(y0_new)
#x1_new = np.array(x1_new)
#y1_new = np.array(y1_new)

#print(x1_new.shape)
#print(y1_new.shape)

#print(x0_new.shape)
#print(y0_new.shape)

#x1_new = np.vstack((x1_new,x0_new))
#y1_new = np.vstack((y1_new,y0_new))
#del x0_new
#del y0_new
#print(x1_new.shape)
#print(y1_new.shape)
#m,n = x1_new.shape
#indx = list(range(m))
#random.shuffle(indx)
#x = [x1_new[i] for i in indx]
#y = [y1_new[i] for i in indx]
#del x1_new
#del y1_new
x = np.vstack((x0,x1))
y = np.vstack((y0,y1))

x = np.array(x)
y = np.array(y)
x, y = shuffle(x,y,random_state=0)
print("X shape ",x.shape)
print("Y shape ",y.shape)

#cross_vald_lin()
#cross_vald_gauss()

if sys.argv[1]=='l':
    print("Linear kernel")
    lin()

elif sys.argv[1]=='g':
    print("Guassian kernel")
    gau()

else:
    print("INCORRECT OPTION")