import numpy as np


def one_hot_en(a,units):
	ar = np.zeros((units,1))
	ar[int(a)]=1.0
	return ar 

def ip_format(data_x):
	return np.array([i.reshape(-1,1) for i in data_x])

def op_format(data_y,units):
	return np.array([one_hot_en(a,units) for a in data_y])

#x1w11+x2w21
def comp_net(x,w,b):
	return np.matmul(w,x)+b

def zero_arr(a,b):
	return np.zeros(a.shape),np.zeros(b.shape)

def sig(x):
	l = (1.0+np.exp(-x))
	return 1.0/l

def dif_sig(o):
	return o*(1-o)

def rlu(x):
	return x*(x>0)

def dif_rlu(x):
	return 1*(x>0)

def cost(x,y):
	l = (np.linalg.norm(x-y)**2)
	return 0.5*l

def accuracy(actual, predicted):
	correct = sum(a == p for (a, p) in zip(actual, predicted))
	return correct / float(len(actual))

class NeuralNet():
	def __init__(self,units,relu=False):
		self.wgts,self.bias = [],[]
		#self.bias = []
		for i,j in zip(units[:-1],units[1:]):
			#w = np.random.randn(j,i)/np.sqrt(i)#array of i*j
			w = np.random.uniform(-0.5,0.5,(j,i))
			self.wgts.append(w)
			b = np.random.randn(j,1)
			self.bias.append(b)
		self.units = units
		self.rlu = relu
		print "Relu ",self.rlu
		self.layers = len(units)
	
	#Gadients at each layer
	def back_prop(self,out,tar):
		dw,db = [0]*(self.layers-1),[0]*(self.layers-1)
		for i in range(1,self.layers):
			if self.rlu and i!=1:
				d_out = dif_rlu(out[-i])
			else:
				d_out = dif_sig(out[-i])
			if not (i==1):			
				wgt = np.transpose(self.wgts[-i+1])
				delt = d_out * np.matmul(wgt,delt)
			else:
				delt = d_out * (out[-1]-tar)
			db[-i] = delt
			ot = np.transpose(out[-i-1])
			dw[-i] = np.matmul(delt,ot)
		return dw,db
			
	#outputs of each layer
	def feed_frwd(self,x,flag=False):
		out = [x]
		for i in range(self.layers-1):
			w,b = self.wgts[i],self.bias[i]
			net_i = comp_net(x,w,b) #wx+b
			if self.rlu and i<(self.layers-2):
				x = rlu(net_i)
			else:
				x = sig(net_i)
			out.append(x)
		if flag:
			return out[-1]
		else:
			return out
	
	
	def grad_des(self,data_x,ym,bat_s=100,eta_f=False):
		x = ip_format(data_x) 		
		ind = np.array([i for i in range(len(x))])
		if self.units[-1]>1:
			y = op_format(ym,self.units[-1])
		epoch = 0
		eta = 0.1
		err = np.inf
		epocs = 1000
		err_th = 10**(-9)
		tol = 10**(-4)
		it = 0
		while True:			
			np.random.shuffle(ind)
			epoch = epoch+1
			print "epoch: ",epoch
			for i in range(0,len(x),bat_s):
				bat = ind[i:i+bat_s]
				dw = [0]*self.layers
				db = [0]*self.layers
				x_b = x[bat]
				y_b = y[bat]
				l = len(x_b)
				for xb,yb in zip(x_b,y_b):
					lay_op = self.feed_frwd(xb)
					grad = 	self.back_prop(lay_op,yb)		
					for i,(w_i,b_i) in enumerate(zip(*grad)):
						if dw[i] is 0:
							dw[i],db[i] = zero_arr(self.wgts[i],self.bias[i])
						dw[i] = dw[i] + (w_i)
						db[i] = db[i] + (b_i)
						
				for m in range(self.layers-1):		
					self.wgts[m] = self.wgts[m]-(eta*(dw[m]/l))
					self.bias[m] = self.bias[m]-(eta*(db[m]/l))
				
				
			err_old = err
			err = self.comp_err(x,y)
			print "Error ",err
			sc = self.score(x,ym)
			print "Score ",sc
			if eta_f and (err_old-err)>tol:
				it = it+1
				if it==2:
					eta = eta/5
					it = 0
			elif eta_f and it==1:
				it = 0
			
			if epoch==epocs:
				print "Error ",err
				print "Epochs ",epoch
				print "Reached maximum epochs "
				break
			elif abs(err_old-err)<=err_th:
				print "Error ",err
				print "Epochs ",epoch
				print "Reached error threshold "
				break
				


	def comp_err(self,x,y):
		l = len(x)
		m = []
		for i,j in zip(x,y):
			c = self.feed_frwd(i,True)
			cs = cost(c,j)
			m.append(cs)
			#print c,j
		#m = [cost(self.feed_frwd(i,True),j) for i,j in zip(x,y)]
		return sum(m)/l
	
	 
	def pred(self,x):
		if self.units[-1] == 1:
			m = [int(self.feed_frwd(a.reshape(-1,1),True)>0.5) for a in x]
			return np.array(m)
		else:
			m = [self.feed_frwd(a.reshape(-1,1),True).argmax() for a in x]
			return np.array(m) 	

	def score(self,x,y):
		a = accuracy(y,self.pred(x))
		return a
