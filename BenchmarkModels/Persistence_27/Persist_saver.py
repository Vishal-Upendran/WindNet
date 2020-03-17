import numpy as np
from glob import glob
import os
from itertools import izip
import shutil
import sys

Tpersist=27

channelNo= int(sys.argv[1])
#Script to split data into 5 to perform cross validation.
path = 'CrossValidation'+str(channelNo)+'/'
p1 = path+'Fold1/'
p2 = path + 'Fold2/'
p3 = path + 'Fold3/'
p4 = path + 'Fold4/'
p5 = path + 'Fold5/'
if not os.path.isdir(path):
	os.makedirs(p1)
	os.makedirs(p2)
	os.makedirs(p3)
	os.makedirs(p4)
	os.makedirs(p5)
else:
	shutil.rmtree(path)
	os.makedirs(p1)
	os.makedirs(p2)
	os.makedirs(p3)
	os.makedirs(p4)
	os.makedirs(p5)
def decisionMaker():
	'''
		Function to decide if the given batch should go to test or training set.
		Idea is to draw a sample i from a uniform distribution of [0,1.0].
		If i<=0.8 => assign to training set.
		Else, assign to test set.
	'''
	i = np.random.uniform(0.0,1.0,1)
	if i<=0.2:
		return p1
	elif i<=0.4:
		return p2
	elif i<=0.6:
		return p3
	elif i<=0.8:
		return p4
	else:
		return p5

#11 and 13 I must
j=1
#for p in ['11','12','13','14','15','16','17','18']:
#tmp = np.loadtxt('../DataProcessing/'+str(channelNo)+"OMNI_20"+p+".txt")
#	print tmp.shape
#	try:
#		dout=np.concatenate([dout,tmp],axis=0)
#	except:
#		dout=tmp
tmp=np.load('OMNIWEB_data_dict.npy',allow_pickle=True).tolist()
swv=tmp['SW Plasma Speed'].reshape([-1,1])
swvstd=tmp['sigma-V'].reshape([-1,1])
dout=np.concatenate([swv,swvstd],axis=1)
#dout=np.reshape(dout,[-1,1])
print dout.shape 
length = dout.shape[0]
din=dout[:length-Tpersist,:]
dout=dout[Tpersist:,:]

for i in np.arange(20,length-Tpersist,20):
	data = {'input': din[i-20:i,:], 'output': dout[i-20:i] }
	bpath = decisionMaker()
	np.save(bpath+'Partition'+str(j),data)
	print data['output'].shape
	print data['input'].shape
	j=j+1
if dout[length-20:].shape[0]==20:
	data = {'input': din[length-20:,:], 'output': dout[length-20:] }
	print data['output'].shape
	print data['input'].shape
	bpath = decisionMaker()
	np.save(bpath+'Partition'+str(j),data)
	j=j+1

print j
print "---------------------------"
