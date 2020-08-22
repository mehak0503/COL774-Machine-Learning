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


#folder = '/home/gem2/media_filter/not useful/validation_dataset'
folder='/home/meh/Music/ML/a4/vald'
lst = sorted(os.listdir(folder))

with open("pca.pkl",'rb') as fil:
	pca = pickle.load(fil)

val_y = np.array([])
val_x = []
for i in range(len(lst)):
    if (lst[i].endswith('.csv')):
        continue
    res = []
    ls = sorted(os.listdir(folder+'/'+lst[i]))
    l = len(ls)
    fold = folder+'/'+lst[i]
    for j in range(len(ls)):
        if (ls[j].endswith('.csv')):
            continue
        else:
            im = Image.open(fold+'/'+ls[j])
            im_ar = np.array(im)
            im_ar = rgb2gray(im_ar)
            res.append(im_ar.flatten()/255)
    res = np.array(res,dtype='float32')
    res = pca.transform(res)
    print(res.shape)

    m,n = res.shape
    x = np.array([])
    for j in range(m):
        x = np.append(x,res[j,:])
    val_x.append(x)
    print("Episode ",i)
val_x = np.array(val_x,dtype='float32')
print("Valid shape ",val_x.shape)
with open('valid_x.csv','ab') as abc:
    np.savetxt(abc,val_x, delimiter=",",fmt="%0.5f")
    


