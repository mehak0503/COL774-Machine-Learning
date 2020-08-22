from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from random import randint
import confmat
import utils
import nltk
import collections
import pickle
import timeit
import math
import re
import sys
cnt = 0
size_voc = 0
doc_data = []
test_data = []
vocab = set()
class_dict = [[[],[],0],[[],[],0],[[],[],0],[[],[],0],[[],[],0]]
class_words = [[],[],[],[],[]]
class_len = [0]*5
class_docs = [[],[],[],[],[]]
prior = [None]*5

def read_train(filename,stem=False,bigram=False):
	global doc_data,cnt,size_voc
	count = [0]*5
	cnt = 0	
	for i in range(5):
		class_dict[i][0] = Counter()
	for doc in utils.json_reader(filename):
		#txt = word_tokenize(doc["text"].lower())		
		txt = re.findall(r"[\w']+", doc["text"].lower())
		#txt = doc["text"].split(" ")
		if stem == True:
			s = " "
			txt = utils.getStemmedDocuments(s.join(txt))
			txt = [item for item in txt if not item.isdigit()]
		if bigram == True:
			txt = (list(nltk.bigrams(txt)))
		vocab.update(txt)
		cnt = cnt+1
		class_dict[int(doc["stars"])-1][0].update(txt)
		class_dict[int(doc["stars"])-1][2] += 1
		doc_data.append([doc["stars"],Counter(txt)])
		
	for i in range(5):
		class_dict[i][1] = sum(class_dict[i][0].values())
		print class_dict[i][1],class_dict[i][2]
	print "vocab"
	print len(vocab)
	size_voc = len(vocab)
	#with open("doc_data.pkl","wb") as pickle_out:
        #        pickle.dump(doc_data,pickle_out)
	#with open("vocab.pkl","wb") as pickle_out:
        #        pickle.dump(class_dict,pickle_out)
	
	#with open("test_data.pkl","wb") as pickle_out:
        #        pickle.dump(doc_data,pickle_out)

def read_test(filename,stem=False,bigram=False):
	global test_data
	for doc in utils.json_reader(filename):
		#txt = word_tokenize(doc["text"].lower())
		txt = re.findall(r"[\w']+", doc["text"].lower())
		#txt = doc["text"].split(" ")
		if stem == True:
			s = " "
			txt = utils.getStemmedDocuments(s.join(txt))
			txt = [item for item in txt if not item.isdigit()]
		if bigram == True:
			txt = (list(nltk.bigrams(txt)))		
		test_data.append([doc["stars"],Counter(txt)])


def read_t(filename,stem=False,bigram=False):
	global doc_data
	for doc in utils.json_reader(filename):
		#txt = word_tokenize(doc["text"].lower())
		txt = re.findall(r"[\w']+", doc["text"].lower())
		#txt = doc["text"].split(" ")
		if stem == True:
			s = " "
			txt = utils.getStemmedDocuments(s.join(txt))
			txt = [item for item in txt if not item.isdigit()]
		if bigram == True:
			txt = (list(nltk.bigrams(txt)))		
		doc_data.append([doc["stars"],Counter(txt)])



def priors():
	global prior
	for i in range(5):
		prior[i] = class_dict[i][2]/float(len(doc_data))

def pred(s,c_flag = False,test=False,tf_idf = False):
	global cnt
	if test==True:
		cnt = len(test_data)
	else:
		cnt = len(doc_data)
	if tf_idf==True:
		d_f,n = compute_idf()
	p = [None]*cnt
	acc_pred = 0
	for i in range(cnt):
		temp = [0]*5
		if test==True:
			temp_d = test_data[i][1]
		else:
			temp_d = doc_data[i][1]
		for k in temp_d:
			for j in range(5): 
				if tf_idf==True and k in class_dict[j][0] :
					phi = ((class_dict[j][0][k]*math.log(n/d_f[k]))+1)/(float(class_dict[j][1])+float(size_voc))

				elif k in class_dict[j][0]: 
					phi = (class_dict[j][0][k]+1)/(float(class_dict[j][1])+float(size_voc))
				else:
					phi = 1/float(size_voc)
				temp[j] = temp[j]+ temp_d[k]*math.log(phi)  
		for j in range(5):
			temp[j] = temp[j]+math.log(prior[j])
		pred = temp.index(max(temp))
		p[i] = pred+1
		st = ""
		if test==True:
			if (pred+1) == int(test_data[i][0]):
				acc_pred = acc_pred + 1
				st = "For i = "+str(i)+" prediction "+str(pred+1)+" test_data "+str(test_data[i][0])
		else:		
			if (pred+1) == int(doc_data[i][0]):
				acc_pred = acc_pred + 1
				st = "For i = "+str(i)+" prediction "+str(pred+1)+" doc_data "+str(doc_data[i][0])
		print(st)

	print "Accuracy over "+str(s)+" is: "
	print((acc_pred/float(cnt))*100)
	f_macro(p,s,test)	
	if c_flag==True:
		conf_matrix(p)


def f_macro(y_pred,s,test=False):
	if test==True:
		true_val = [int(test_data[i][0]) for i in range(len(test_data))]
	else:
		true_val = [int(doc_data[i][0]) for i in range(len(doc_data))]
	f_score = f1_score(true_val, y_pred, average=None)
	print "F1_score for "+str(s)+" is: "
	print f_score
	f_mac = f1_score(true_val, y_pred, average='macro')  
	print "F1_macro_score for "+str(s)+" is: "
	print f_mac
	
	
def random_pred():
	cnt = len(test_data)
	p = [None]*cnt
	acc_pred = 0
	for i in range(cnt):
		pred = randint(1, 5)
		p[i] = pred
		if pred == int(test_data[i][0]):
			acc_pred = acc_pred + 1
	
	print("Accuracy for random prediction is: ")
	print((acc_pred/float(cnt))*100)
	f_macro(p,"random prediction",True)	

def majority_pred():
	cnt = len(test_data)
	maj_doc = [0]*5
	for it in doc_data:
		maj_doc[int(it[0])-1] +=1
	doc_maj = 0
	max_class = maj_doc.index(max(maj_doc)) + 1	
	for i in range(cnt):
		if int(test_data[i][0])==max_class:
			doc_maj = doc_maj+1
	print("Accuracy for majority prediction is: ")
	print((doc_maj/float(cnt))*100)
	f_macro([max_class]*cnt,"majority prediction",True)	

def conf_matrix(p):
	cnt = len(test_data)
	actual = [int(test_data[i][0]) for i in range(cnt)]
	pred = [int(p[i]+1) for i in range(cnt)]
	con_mat = confusion_matrix(actual,pred)
	print("Confusion matrix on test set is: ")
	print(con_mat)
	confmat.plot_confusion(con_mat,list(set(actual)),"Naive Bayes Prediction on Test Data")	
	
def compute_idf():
	n = len(doc_data)
	d_f = Counter()
	for it in doc_data:
		d_f.update(set(it[1].keys()))
	return d_f,n
				



def part_a(f_train,f_test,flag=False):
	read_train(f_train)
	priors()
	pred(" training set in part a ")			
        read_test(f_test)
	pred(" testing set in part a ",True,flag)

def part_b(f_train,f_test):
	read_t(f_train)
	read_test(f_test)
	random_pred()
	majority_pred()

def part_c(f_train,f_test):
	part_a(f_train,f_test,True)

def part_d(f_train,f_test):
	read_train(f_train,True)
	priors()
	pred(" training set with stemming ")			
        read_test(f_test,True)
	pred(" testing set with stemming ",False,True)
		
def part_ea(f_train,f_test):
	read_train(f_train,True,True)
	priors()
	pred(" training set with stemming and bigrams ")			
        read_test(f_test,True,True)
	pred(" testing set with stemming and bigrams ",False,True)
	
	
def part_eb(f_train,f_test):
	read_train(f_train,True)
	priors()
	pred(" training set with stemming and tf_idf ",False,False,True)			
        read_test(f_test,True)
	pred(" testing set with stemming and tf_idf ",False,True,True)
	
def part_g(f_train,f_test):
	read_train(f_train,True,True)	
	priors()			
        read_test(f_test,True,True)
	pred(" testing set with stemming and bigrams",False,True)

if(len(sys.argv)<4):
	print('Insufficient arguments')
	sys.exit()

if sys.argv[3] == 'a':
	print "a"
	start = timeit.default_timer()	
	part_a(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part a takes ",stop-start,"seconds"
	
elif sys.argv[3] == 'b':
	print "b"
	start = timeit.default_timer()	
	part_b(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part b takes ",stop-start,"seconds"

elif sys.argv[3] == 'c':	
	print "c"
	start = timeit.default_timer()	
	part_c(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part c takes ",stop-start,"seconds"

elif sys.argv[3] == 'd':
	print "d"
	start = timeit.default_timer()	
	part_d(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part d takes ",stop-start,"seconds"

elif sys.argv[3] == 'e':
	print "e"
	start = timeit.default_timer()	
	part_ea(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part ea takes ",stop-start,"seconds"
	start = timeit.default_timer()	
	part_eb(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part eb takes ",stop-start,"seconds"

elif sys.argv[3] == 'g':
	print "g"
	start = timeit.default_timer()	
	part_g(sys.argv[1],sys.argv[2])
	stop = timeit.default_timer()
	print "Part g takes ",stop-start,"seconds"

else:
	print "Incorrect part number"
