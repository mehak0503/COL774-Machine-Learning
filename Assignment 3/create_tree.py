import numpy as np
import queue
from tqdm import tqdm
from functools import partial
from collections import defaultdict

cat = [2,3,4,6,7,8,9,10,11]
#return partitions
def parts(x,binry=True):
	if not binry:
		med = np.median(x)
		x = (x>=med).astype(int)
	attr_dict = {key: np.where(x == key)[0] for key in np.unique(x)}
	return attr_dict
			
def binary_attr():
	global binary
	for i in range(24):
		if i not in cat:
			binary[i] = 1
	

binary = np.zeros((25-1,1))
binary_attr()
cat_attr = [i for i in binary if i]

def leng(l):
	return len(l)

def get_best(a):
	return a.index(max(a))	

def update_acc(acc,s_train,s_test,s_vald):
	acc["train"].append(s_train)
	acc["test"].append(s_test)
	acc["validation"].append(s_vald)
	return acc

def acc_mtx():
	return {"train":[],"validation":[],"test":[]}

def get_cnt(x):
	return np.bincount(x)

def accuracy(actual, predicted):
	correct = sum(a == p for (a, p) in zip(actual, predicted))
	return correct / float(len(actual))


def entropy(y):
	prob = np.bincount(y)/float(len(y))
	summ = sum(float(p)*float(np.log2(p)) for p in prob if p)
	return -1*summ

def cond_entropy(part,y,xval):
	summ = 0.0
	for p in part.values():
		summ = summ + (float(len(p))/len(xval))*entropy(y[p])
	return summ
	
def info_gain(y,ent):
	return entropy(y)-ent
class Node():
	def __init__(self,parent,noofsamples,spl_attr = None,children=[]):
		self.noofsamples = noofsamples
		self.parent = parent
		self.med = None
		self.clas = noofsamples.argmax()
		self.spl = [spl_attr,None]
		self.children = children


class dtree():
	def __init__(self,train_set=None,unwght = False):
		if train_set is None:
			self.unwgt = unwght		
			self.root = None
		else:
			self.unwgt = unwght
			self.root = self.build_tree(train_set)	

	
	def score(self,data_set):
		predicts = [self.predict_val(self.root,x) for x in data_set]
		return 100*accuracy(data_set[:,-1],predicts)

	
	def noofnodes(self):   
		return self.no_of_nodes(self.root)


	def height(self):
		return self.hgt(self.root)	

	
	def select_attr(self,train_set):
		max_gain = -1
		attr = -1
		y = train_set[:,-1]
		x = train_set[:,1:23]
		for i,xval in enumerate(np.transpose(x)):
			if self.unwgt and binary[i+1]:
				part = parts(xval,False)
			else:
				part = parts(xval)
			entropy_y_xval = cond_entropy(part,y,xval)
			gain = info_gain(y,entropy_y_xval)
			if gain > max_gain:
				max_gain = gain
				attr = i+1 #as first column is index
		return [max_gain, attr]

	def build_tree(self,train_set,parent = None):
		n_samp = get_cnt(train_set[:,-1])
		s = set(train_set[:,-1])
		if len(s) <= 1:
			return Node(parent,n_samp)

		l = self.select_attr(train_set)
		if l[0] > 0:
			node = Node(parent,n_samp,l[1],children=[])
			if self.unwgt and binary[l[1]]:
				partitions = parts(train_set[:,l[1]],False).items()
				node.med = np.median(train_set[:,l[1]])
				if len(partitions)==1:
					return node
			else:
				partitions = parts(train_set[:,l[1]]).items()
			

			for val,ind in partitions:
				tt = train_set[ind]
				child = self.build_tree(tt,parent=node)
				child.spl[1] = val
				node.children.append(child)
			return node
		else:
			return Node(parent,n_samp)

	@staticmethod
	def hgt(d_tree):			
		if not d_tree.children:
			return 0
		else:
			return 1+max(map(dtree.hgt,d_tree.children))

	def attr_thresh(self):
		d = {}
		for i in range(25-2):
			if binary[i]==1:
				d[i] = 	self.n_att_th(self.root,i)
		return d

	def predict_val(self,d_tree,x):
		if not d_tree.children:
			return d_tree.clas			
		else:
			x_val = x[d_tree.spl[0]]
			childr = {}
			if d_tree.med is not None:
				x_val = int(x_val>=d_tree.med) 
			for ch in d_tree.children:
				childr[ch.spl[1]] = ch
			child = childr.get(x_val)
			if not child:
				return d_tree.clas
			else:
				return self.predict_val(child,x)

	
	@staticmethod	
	def no_of_nodes(d_tree):
		if not d_tree.children:
			return 1
		else:
			return 1+sum(map(dtree.no_of_nodes,d_tree.children))

	def getnodes(self):
		q = queue.Queue(0)
		q.put(self.root)
		l = []
		while not (q.empty()):
			n = q.get()
			l.append(n)
			if n.children:
				for i in n.children:
					q.put(i)
		return l

	
	def rem_node(self,n):
		p = n.parent
		if p is None:
			return
		p.children.remove(n)
		return p



	@staticmethod	
	def n_att_th(d_tree,attr):
		if not d_tree.children:
			return []
		else:
			m = partial(dtree.n_att_th,attr=attr)
			f1 = (attr==d_tree.spl[0])
			max_th = max(map(m,d_tree.children),key=len)
			if f1 and d_tree.med is not None:
				return [d_tree.med] + max_th
			else:
				return list(max_th)
			

	def pruning(self,train,test,vald):
		acc = acc_mtx()
		t_nodes = self.getnodes()
		n_cnt = []
		t_nodes.reverse()
		for n in tqdm(t_nodes,ncols=80,ascii=True):
			if not n.children:
				continue
			child_bkp = n.children
			acc_bef_prune = self.score(vald)
			n.children = []
			acc_aft_prune = self.score(vald)
			if acc_bef_prune > acc_aft_prune:
				n.children = child_bkp
			elif acc_bef_prune < acc_aft_prune:				
				n_cnt.append(self.noofnodes())
				acc = update_acc(acc,self.score(train),self.score(test),acc_aft_prune)
		return n_cnt,acc
	
	
