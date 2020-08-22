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

#folder = '/home/ictd/Music/validation_dataset'
folder='/home/meh/Music/ML/a4'

lst = sorted(os.listdir(folder))


for i in range(len(lst)):
    val_x = []
    res = []
    ls = sorted(os.listdir(folder+'/'+lst[i]))
    l = len(ls)
    fold = folder+'/'+lst[i]
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
    x = []
    for j in range(m):
        x.append(res[j,:].reshape(210,160))
    x = np.stack(x,axis=2)
    val_x.append(x)
    print("Episode ",i)
    if i/300==1:
        form_zip(np.array(val_x),str(i-300))
        val_x=[]

    