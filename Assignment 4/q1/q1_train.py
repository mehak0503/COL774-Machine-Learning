import numpy as np
import pandas as pd
from PIL import Image
import os
import random
from sklearn.decomposition import PCA
import pickle
from itertools import combinations
from collections import Counter
import timeit

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


folder = '/home/gem2/media_filter/not useful/my'
#folder='/home/meh/Music/ML/a4'
lst = sorted(os.listdir(folder))

with open("pca.pkl",'rb') as fil:
	pca = pickle.load(fil)

for i in range(len(lst)):
    train_x_0 = []
    train_x_1 = []
    res = []
    ls = sorted(os.listdir(folder+'/'+lst[i]))
    #ls = sorted(os.listdir(folder+'/00000001'))
    l = len(ls)
    fold = folder+'/'+lst[i]
    #fold = folder+'/00000001'
    rew = np.array([])
    for j in range(len(ls)):
        if (ls[j].endswith('.csv')):
            tmp = pd.read_csv(fold+'/'+ls[j],header=None)
            tmp = tmp.iloc[:,:].values
            rew = tmp.flatten()
            rew = rew.reshape(-1,1)
            continue
        else:
            im = Image.open(fold+'/'+ls[j])
            im_ar = np.array(im)
            im_ar = rgb2gray(im_ar)
            res.append(im_ar.flatten()/255)
    res = np.array(res,dtype='float32')
    res = pca.transform(res)
    print(res.shape)
    cntr = 0
    m,n = res.shape
    for j in range(m):
        if j+7 >= m-1:
            break
        if int(rew[j+7,:])==1:
            for itr in list(combinations(range(j,j+6),4)):
                x = np.array([])
                for it in itr:
                    x = np.append(x,res[it,:])
                x = np.append(x,res[j+6,:])
                train_x_1.append(x)
            cntr = cntr+2
        elif cntr>0:
            for itr in list(combinations(range(j,j+6),4)):
                x = np.array([])
                for it in itr:
                    x = np.append(x,res[it,:])
                x = np.append(x,res[j+6,:])
                train_x_0.append(x)
            cntr = cntr-1
        
    print(Counter(rew.flatten().astype(int)))
    print("Episode ",i)

    train_x_1 = np.array(train_x_1,dtype='float32')
    print("Train shape 1 ",train_x_1.shape)
    
    with open('train_x_1.csv','ab') as abc:
        np.savetxt(abc,train_x_1, delimiter=",",fmt="%0.5f")
    
    train_x_0 = np.array(train_x_0,dtype='float32')
    
    print("Train shape 0 ",train_x_0.shape)
    with open('train_x_0.csv','ab') as abc:
        np.savetxt(abc,train_x_0, delimiter=",",fmt="%0.5f")
    
    