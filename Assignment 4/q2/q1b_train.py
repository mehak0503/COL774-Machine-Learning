import numpy as np
import pandas as pd
from PIL import Image
import os
import random
import pickle
from itertools import combinations
from collections import Counter
import timeit
import _pickle
import gzip

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def form_zip(train_set,name):
    with gzip.open(name,'wb') as fil:
        _pickle.dump(train_set,fil)

#folder = '/home/ictd/Music/data'
folder='/home/meh/Music/ML/a4'
lst = sorted(os.listdir(folder))


#for i in range(len(lst)):
for i in range(1):
    train_x_0 = []
    train_x_1 = []
    res = []
    #ls = sorted(os.listdir(folder+'/'+lst[i]))
    ls = sorted(os.listdir(folder+'/00000001'))
    l = len(ls)
    #fold = folder+'/'+lst[i]
    fold = folder+'/00000001'
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
            res.append(im_ar.flatten())
    res = np.array(res,dtype='float32')
    print(res.shape)
    m,n = res.shape
    cnt = 0
    for j in range(m):
        if j+7 >= m-1:
            break
        if int(rew[j+7,:])==1:
            for itr in list(combinations(range(j,j+6),4)):
                x = []
                for it in itr:
                    x.append(res[it,:].reshape(210,160))
                x.append(res[j+6,:].reshape(210,160))
                x = np.stack(x,axis=2)
                train_x_1.append(x)
            cnt = cnt+2
        elif cnt>0:
            for itr in list(combinations(range(j,j+6),4)):
                x = []
                for it in itr:
                    x.append(res[it,:].reshape(210,160))
                x.append(res[j+6,:].reshape(210,160))
                x = np.stack(x,axis=2)
                train_x_0.append(x)
            cnt = cnt-1
    print(Counter(rew.flatten().astype(int)))
    print("Episode ",i)


    train_x_1 = np.array(train_x_1,dtype='float32')
    print("Train shape 1 ",train_x_1.shape)  
    train_x_0 = np.array(train_x_0,dtype='float32')
    print("Train shape 0 ",train_x_0.shape)
    z = np.array([train_x_0,train_x_1])
    form_zip(z,str(lst[i]))

    