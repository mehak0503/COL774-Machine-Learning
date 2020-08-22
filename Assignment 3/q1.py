import sys
import numpy as np
from numpy import genfromtxt
from create_tree import dtree
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import timeit
cat = [2,3,4,6,7,8,9,10,11]

def leng(l):
	return len(l)

def par():
	param = {'max_depth':range(5,25),
		 'min_samples_split':range(5,100,5),
		 'min_samples_leaf':range(5,75,5)
		}
	return param

def parm():
	param = {'n_estimators':range(5,25),
		 'max_features':range(3,15),
		 'max_depth':range(5,15),
		 'bootstrap':[True,False]
		}
	return param

def get_best(a,flag):
	if flag:
		return a.index(min(a))
	else:
		return a.index(max(a))	

def sk_class(dep=None,sam_spl=2,sam_leaf=1):
	return DecisionTreeClassifier(criterion="entropy",random_state=0,
			max_depth=dep,min_samples_split=sam_spl,min_samples_leaf=sam_leaf)

def rd_forest():
	return RandomForestClassifier(criterion="entropy",random_state=0)

def leg(acc,pos):
	return ['Train: '+str(round(acc["train"][pos],1)),
		'Validation: '+str(round(acc["validation"][pos],1)),
		'Test: '+str(round(acc["test"][pos],1))] 

def get_n(d_tree):
	n = d_tree.getnodes()
	n.reverse()
	return n

def lab_enc(d):
	le = LabelEncoder()
	int_e = le.fit_transform(d)
	int_e = int_e.reshape(d.size,1)
	return int_e

def hot_enc(d):
	oe = OneHotEncoder(sparse=False)
	encd = oe.fit_transform(d)
	return encd

def plt_save(n_cnt,acc,s,flag=False):
	plt.plot(n_cnt,acc["train"])
	plt.plot(n_cnt,acc["validation"])
	plt.plot(n_cnt,acc["test"])
	plt.xlabel('No of nodes')
	plt.ylabel('Accuracy')	
	pos = get_best(n_cnt,flag)
	if flag:
		plt.gca().invert_xaxis()	
	plt.legend(leg(acc,pos))
	plt.savefig(s+'.png')
	plt.show()

def update_acc(acc,s_train,s_test,s_vald):
	acc["train"].append(s_train)
	acc["test"].append(s_test)
	acc["validation"].append(s_vald)
	return acc

def acc_mtx():
	return {"train":[],"validation":[],"test":[]}

def binary_attr():
	global binary
	binary[0] = 1
	binary[1] = 1
	binary[5] = 1
	for i in range(n-1-12):
		binary[i+12] = 1

def comp_median(arr):
	return np.median(arr,axis=0)

def plot_acc(d_tree,s):
	acc = acc_mtx()
	t_nodes = get_n(d_tree)
	noofnodes = leng(t_nodes)
	n_cnt = []
	for i in tqdm(range(0,leng(t_nodes),100),ncols=80,ascii=True):
		for n in t_nodes[i:i+100]:
			d_tree.rem_node(n)
		noofnodes = max(noofnodes-100,0)
		n_cnt.append(noofnodes)
		if s=='a':
			acc = update_acc(acc,d_tree.score(x1),d_tree.score(test1),d_tree.score(vald1))
		elif s=='c':
			acc = update_acc(acc,d_tree.score(x),d_tree.score(tests),d_tree.score(vald))
	plt_save(n_cnt,acc,s)
	
def pre_process(x,m,n):
	medians = [0]*(n-1)
	for i in range(n-1):
		if binary[i]==0:
			continue
		medians[i] = comp_median(x[:,i])
		for j in range(m):
			if x[j,i] >= medians[i]:
				x[j,i] = 1
			else:
				x[j,i] = 0
	return x

def pre_p():
	x1 = x,vald1 = vald,test1 = tests
	x_set = pre_process(x1,m,n)	
	vald_set = pre_process(vald1,mv,nv)
	test_set = pre_process(test1,mt,nt)
	return x_set,vald_set,test_set

def part_a():
	d_tree = dtree(x1)
	print "Tree height ",d_tree.height()
	print "Number of nodes ",d_tree.noofnodes()
	print "Accuracy (training set) ",d_tree.score(x1)
	print "Accuracy (validation set) ",d_tree.score(vald1)
	print "Accuracy (testing set) ",d_tree.score(test1)
	plot_acc(d_tree,'a')	

def part_b():
	d_tree = dtree(x1)
	n_cnt,acc=d_tree.pruning(x1,test1,vald1)
	print "Tree height ",d_tree.height()
	print "Number of nodes ",d_tree.noofnodes()
	print "Accuracy (training set) ",d_tree.score(x1)
	print "Accuracy (validation set) ",d_tree.score(vald1)
	print "Accuracy (testing set) ",d_tree.score(test1)
	plt_save(n_cnt,acc,'b',True)

def part_c():
	d_tree = dtree(x,True)
	print "Tree height ",d_tree.height()
	print "Number of nodes ",d_tree.noofnodes()
	print "Accuracy (training set) ",d_tree.score(x)
	print "Accuracy (validation set) ",d_tree.score(vald)
	print "Accuracy (testing set) ",d_tree.score(tests)
	for att,th in d_tree.attr_thresh().items():
		if len(th)>1:
			print att,th		
	plot_acc(d_tree,'c')	
		
	
def plotting(x,y,label):
	plt.plot(x,y,linestyle='--',marker='o')
	plt.xlabel(label)
	plt.ylabel("Validation accuracy")
	plt.savefig(label+".png")
	plt.show()

def grid_srch():
	start = timeit.default_timer()
	res = {}
	param = parm()
	for p in ParameterGrid(param):
		classifier = RandomForestClassifier(criterion="entropy",random_state=0,**p)
		classifier.fit(x[:,1:n-2],x[:,-1])
		v_score = classifier.score(vald[:,1:n-2],vald[:,-1])
		res[v_score] = p
		
	m = max(res)
	classifier = RandomForestClassifier(criterion="entropy",random_state=0,**(res[m]))
	classifier.fit(x[:,1:n-2],x[:,-1])
	tr_score = classifier.score(x[:,1:n-2],x[:,-1])
	ts_score = classifier.score(tests[:,1:n-2],tests[:,-1])
	print "Parameters : ",res[m]
	print "Train set accuracy: ",100*tr_score
	print "Validation set accuracy: ",100*m
	print "Test set accuracy: ",100*ts_score
	stop = timeit.default_timer()
	print "Execution time ",(stop-start)

	

def depth_class(p):
	acc = []
	for d in range(2,35):
		classifier = sk_class(d)
		classifier.fit(x[:,1:n-2],x[:,-1])
		acc.append(100*classifier.score(vald[:,1:n-2],vald[:,-1]))
	plotting(range(2,35),acc,"Max_Depth"+p)
	

def min_samp_spl(p):
	acc = []
	for s in range(5,100,5):
		classifier = sk_class(None,s)
		classifier.fit(x[:,1:n-2],x[:,-1])
		acc.append(100*classifier.score(vald[:,1:n-2],vald[:,-1]))
	plotting(range(5,100,5),acc,"Min_Samples_Split"+p)
	
def min_samp_lf(p):
	acc = []
	for l in range(5,100,5):
		classifier = sk_class(None,2,l)
		classifier.fit(x[:,1:n-2],x[:,-1])
		acc.append(100*classifier.score(vald[:,1:n-2],vald[:,-1]))
	plotting(range(5,100,5),acc,"Min_Samples_Leaf"+p)

def param_set():
	start = timeit.default_timer()
	res = {}
	param = par()
	for p in ParameterGrid(param):
		classifier = DecisionTreeClassifier(criterion="entropy",random_state=0,**p)
		classifier.fit(x[:,1:n-2],x[:,-1])
		v_score = classifier.score(vald[:,1:n-2],vald[:,-1])
		res[v_score] = p
	m = max(res)
	classifier = DecisionTreeClassifier(criterion="entropy",random_state=0,**(res[m]))
	classifier.fit(x[:,1:n-2],x[:,-1])
	tr_score = classifier.score(x[:,1:n-2],x[:,-1])
	ts_score = classifier.score(tests[:,1:n-2],tests[:,-1])
	print "Parameters : ",res[m]
	print "Train set accuracy: ",100*tr_score
	print "Validation set accuracy: ",100*m
	print "Test set accuracy: ",100*ts_score
	stop = timeit.default_timer()
	print "Execution time ",(stop-start)

def part_d(p='d'):
	classifier = sk_class()
	classifier.fit(x[:,1:n-2],x[:,-1])
	accur_tr = 100*classifier.score(x[:,1:n-2],x[:,-1])
	accur_v = 100*classifier.score(vald[:,1:n-2],vald[:,-1])
	accur_t = 100*classifier.score(tests[:,1:n-2],tests[:,-1])
	print "Accuracy on training set ",accur_tr
	print "Accuracy on validation set ",accur_v
	print "Accuracy on testing set ",accur_t
	h = classifier.tree_.max_depth
	print "Height of tree ",h
	n_cnt = classifier.tree_.node_count
	print "No of nodes ",n_cnt
	depth_class(p)
	min_samp_spl(p)
	min_samp_lf(p)
	param_set()
	
def data_add():
	global x,vald,tests
	r = x[1,:]
	itr = 0
	for i in cat:
		minn = min(np.amin(x[:,i]),np.amin(vald[:,i]),np.amin(tests[:,i]))
		maxx = max(np.amax(x[:,i]),np.amax(vald[:,i]),np.amax(tests[:,i]))
		for j in range(minn,maxx+1):
			r[i] = j
			x = np.append(x,[r],axis=0)
			vald = np.append(vald,[r],axis=0)
			tests = np.append(tests,[r],axis=0)
			itr = itr+1
	return itr			

def data_rem(rows):
	global x,vald,tests
	x = x[:-rows,:]
	vald = vald[:-rows,:]
	tests = tests[:-rows,:]	


def enc_x():
	global x,m,n
	y = x[:,-1]
	x = np.delete(x,-1,axis=1)
	for i in cat:
		int_encd = lab_enc(x[:,i])
		hot_encd = hot_enc(int_encd)
		x = np.append(x,hot_encd,axis=1)
		
	for i in cat:
		x = np.delete(x,i,axis=1)
	x = np.append(x,y.reshape(y.size,1),axis=1)	
	m,n = x.shape
	

def enc_v():
	global vald,mv,nv
	y = vald[:,-1]
	vald = np.delete(vald,-1,axis=1)
	for i in cat:
		int_encd = lab_enc(vald[:,i])
		hot_encd = hot_enc(int_encd)
		vald = np.append(vald,hot_encd,axis=1)
		
	for i in cat:
		vald = np.delete(vald,i,axis=1)
	vald = np.append(vald,y.reshape(y.size,1),axis=1)	
	mv,nv = vald.shape
	

def enc_t():
	global tests,mt,nt
	y = tests[:,-1]
	tests = np.delete(tests,-1,axis=1)
	for i in cat:
		int_encd = lab_enc(tests[:,i])
		hot_encd = hot_enc(int_encd)
		tests = np.append(tests,hot_encd,axis=1)
		
	for i in cat:
		tests = np.delete(tests,i,axis=1)
	tests = np.append(tests,y.reshape(y.size,1),axis=1)	
	mt,nt = tests.shape
	

def part_e():
	rows = data_add()	
	enc_x()
	enc_v()
	enc_t()	
	data_rem(rows)
	#Using params of previous part
	'''classifier = sk_class(5,95,70)
	classifier.fit(x[:,1:n-2],x[:,-1])
	accur_tr = 100*classifier.score(x[:,1:n-2],x[:,-1])
	accur_v = 100*classifier.score(vald[:,1:n-2],vald[:,-1])
	accur_t = 100*classifier.score(tests[:,1:n-2],tests[:,-1])
	print "Accuracy on training set ",accur_tr
	print "Accuracy on validation set ",accur_v
	print "Accuracy on testing set ",accur_t
	h = classifier.tree_.max_depth
	print "Height of tree ",h
	n_cnt = classifier.tree_.node_count
	print "No of nodes ",n_cnt'''	
	part_d('e')


def n_est():
	acc = []
	for d in range(5,25):
		classifier = RandomForestClassifier(criterion="entropy",random_state=0,n_estimators=d)
		classifier.fit(x[:,1:n-2],x[:,-1])
		acc.append(100*classifier.score(vald[:,1:n-2],vald[:,-1]))
	plotting(range(5,25),acc,"N_Estimators")
	

def max_feat():
	acc = []
	for d in range(3,15):
		classifier = RandomForestClassifier(criterion="entropy",random_state=0,max_features=d)
		classifier.fit(x[:,1:n-2],x[:,-1])
		acc.append(100*classifier.score(vald[:,1:n-2],vald[:,-1]))
	plotting(range(3,15),acc,"Max_Features")
	

def max_dep():
	acc = []
	for d in range(5,15):
		classifier = RandomForestClassifier(criterion="entropy",random_state=0,max_depth=d)
		classifier.fit(x[:,1:n-2],x[:,-1])
		acc.append(100*classifier.score(vald[:,1:n-2],vald[:,-1]))
	plotting(range(5,15),acc,"Max_Depth")
	


def part_f():
	r_for = rd_forest()
	r_for.fit(x[:,1:n-2],x[:,-1])
	accur_tr = 100*r_for.score(x[:,1:n-2],x[:,-1])
	accur_v = 100*r_for.score(vald[:,1:n-2],vald[:,-1])
	accur_t = 100*r_for.score(tests[:,1:n-2],tests[:,-1])
	print "Accuracy on training set ",accur_tr
	print "Accuracy on validation set ",accur_v
	print "Accuracy on testing set ",accur_t
	max_dep()
	max_feat()
	n_est()	
	grid_srch()
	



x = genfromtxt(sys.argv[2],delimiter=',',dtype=int)
#x = genfromtxt('credit-cards.train.csv',delimiter=',',dtype=int)
x = np.delete(x, slice(0,2), axis=0)
m,n = x.shape
binary = np.zeros((n-1,1))
binary_attr()


vald = genfromtxt(sys.argv[4],delimiter=',',dtype=int)
#vald = genfromtxt('credit-cards.val.csv',delimiter=',',dtype=int)
vald = np.delete(vald, slice(0,2), axis=0)
mv,nv = vald.shape

tests = genfromtxt(sys.argv[3],delimiter=',',dtype=int)
#tests = genfromtxt('credit-cards.test.csv',delimiter=',',dtype=int)
tests = np.delete(tests, slice(0,2), axis=0)
mt,nt = tests.shape

x1 = np.copy(x)
vald1 = np.copy(vald)
test1 = np.copy(tests)
pre_process(x1,m,n)	
pre_process(vald1,mv,nv)
pre_process(test1,mt,nt)


if int(sys.argv[1])==1:
	part_a()
elif int(sys.argv[1])==2:
	part_b()
elif int(sys.argv[1])==3:
	part_c()
elif int(sys.argv[1])==4:
	part_d()
elif int(sys.argv[1])==5:
	part_e()
elif int(sys.argv[1])==6:
	part_f()
else:
	print "Incorrect part number"


