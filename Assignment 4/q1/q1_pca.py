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


#folder = '/media/cmkmanwani/hdd/chirag/train_dataset'
folder='/home/meh/Music/ML/a4'

lst = sorted(os.listdir(folder))
train_y = np.array([])
train_x = []
res = []
for ind in range(50):
    ls = sorted(os.listdir(folder+'/'+lst[ind]))
    #ls = sorted(os.listdir(folder+'/00000001'))
    l = len(ls)
    fold = folder+'/'+lst[ind]
    #fold = folder+'/00000001'
    for j in range(len(ls)):
        if not (ls[j].endswith('.png')):
            continue
        im = Image.open(fold+'/'+ls[j])
        a = np.array(im)
        a = rgb2gray(a)
        res.append(a.flatten())
    
res = np.array(res,dtype='float32')
print(res.shape)
pca = PCA(n_components=50)
p_c = pca.fit(res)
with open("pca.pkl","wb") as fil:
    pickle.dump(p_c,fil)

