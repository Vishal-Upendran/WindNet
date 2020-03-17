import numpy as np
from glob import glob
import os
from itertools import izip
import shutil
import sys
from datetime import datetime as dt 
channelNo= int(sys.argv[1])
#Script to split data into 5 to perform cross validation.
j=1
path = '../Data/CrossValidation'+str(channelNo)+'/'
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
def Check_missing(paths):
	name1=paths[0]
	for enumvar,name2 in enumerate(paths[1:]):
		date_n1=dt.strptime(name1.split('_')[-1].split('.')[0],'%Y%m%d').date()
		date_n2=dt.strptime(name2.split('_')[-1].split('.')[0],'%Y%m%d').date()
		if (date_n2-date_n1).days!=1.0:
			return 1,enumvar+1
		else:
			pass

		name1=name2 
	return 0,enumvar


yr_path=sorted(glob(("../Processed_data/"+str(channelNo)+'/_20'+"*.npy")))
din=[]
for yr in yr_path:
	din.append(np.load(yr))
din=np.asarray(din)
for p in ['11','12','13','14','15','16','17','18']:
	tmp = np.loadtxt(str(channelNo)+"OMNI_20"+p+".txt")
	try:
		dout = np.vstack((dout,tmp)) 
	except:
		dout= tmp 
print dout.shape
print din.shape 

length = din.shape[0]
print din.shape,dout.shape

count=0
i=20
#for i in np.arange(20,length,20):
while i<=length:
	"""
		Algo: Consider a batch with 20 points. If there are no gaps in this set, we just save it.
		Else, we find the first discontinuity (let's say at enumvar, within the set). The absolute index
		for this point will be i-20+enumvar (from the start). Now, we select from this point to 20 points
		prior, and save it in the same fold as the previous fold. 

	"""
	#data = {'input': din[i-20:i,:], 'output': dout[i-20:i,:] }
	#bpath = decisionMaker()
	FLAG,enumvar = Check_missing(yr_path[i-20:i])
	#print i
	if FLAG:
		if i==20:
			continue 
		else:
			i=i-20+enumvar
			print "------",yr_path[i]
			data = {'input': din[i-20:i,:], 'output': dout[i-20:i,:] }
			print bpath 
			count+=1
			np.save(bpath+'Partition'+str(j),data)
			print data['output'].shape
			print data['input'].shape
	else:
		data = {'input': din[i-20:i,:], 'output': dout[i-20:i,:] }
		bpath = decisionMaker()
		print bpath 
		np.save(bpath+'Partition'+str(j),data)
		print data['output'].shape
		print data['input'].shape
	#print j
	i+=20
	j=j+1
print count ,j
print dout[i:].shape 


# #11 and 13 I must
# for p in ['11','12','13','14','15','16','17','18']:
# 	yr_path=sorted(glob(("../Processed_data/"+str(channelNo)+'/_20'+p+"*.npy")))
# 	din=[]
# 	for yr in yr_path:
# 		din.append(np.load(yr))
# 	din=np.asarray(din)
# 	dout = np.loadtxt(str(channelNo)+"OMNI_20"+p+".txt")

# 	length = din.shape[0]
# 	print din.shape,dout.shape

# 	for i in np.arange(20,length,20):
# 		data = {'input': din[i-20:i,:], 'output': dout[i-20:i] }
# 		bpath = decisionMaker()
# 		#np.save(bpath+'Partition'+str(j),data)
# 		#print data['output'].shape
# 		#print data['input'].shape
# 		print j
# 		j=j+1
# 	if dout[length-20:].shape[0]==20: #was dout[length-20:]
# 		# MAKE SURE THE PARTITION DONE HERE AND THE JUST PREVIOUS ONE ARE IN THE SAME FOLD.
# 		# ELSE THERE WILL BE A DATA LEAK.
# 		data = {'input': din[i:,:], 'output': dout[i:] }
# 		#print data['output'].shape
# 		#print data['input'].shape
# 		bpath = decisionMaker()
# 		#np.save(bpath+'Partition'+str(j),data)
# 		print "Hello ",j
# 		j=j+1


# 	print j
# 	print str(p)+" done!"
# 	print "---------------------------"

#For creating 3-channel AIA input, use this code.
# for p in ['11','12','13','14','15','16']:
# 	for channelNo in ['bx','by','bz']:
# 		yr_path=sorted(glob(("Processed_data/"+str(channelNo)+'/_20'+p+"*.npy")))
# 		din_tmp=[]
# 		for yr in yr_path:
# 			din_tmp.append(np.load(yr))
# 		din_tmp=np.asarray(din_tmp).reshape([-1,512,512,1]) #HMI has size (512,512). AIA EUV has (256,256). Make sure to change it.
# 		try:
# 			din=np.concatenate((din,din_tmp),axis=-1)
# 		except:
# 			din=din_tmp
# 		print din.shape
# 	dout = np.loadtxt("OMNI_20"+p+".txt")

# 	length = din.shape[0]
# 	print din.shape[0]
# 	print length

# 	for i in np.arange(20,length,20):
# 		data = {'input': din[i-20:i,:], 'output': dout[i-20:i] }
# 		bpath = decisionMaker()
# 		np.save(bpath+'Partition'+str(j),data)
# 		j=j+1
# 	data = {'input': din[length-20:,:], 'output': dout[length-20:] }
# 	bpath = decisionMaker()
# 	np.save(bpath+'Partition'+str(j),data)
# 	j=j+1

# 	print j
# 	print str(p)+" done!"
# 	print "---------------------------"
