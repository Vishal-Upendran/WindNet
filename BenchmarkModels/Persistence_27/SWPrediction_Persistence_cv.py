import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import sys
import os
from sklearn.svm import SVR as Regressor
from sklearn.externals import joblib
sys.path.append('../')
# Custom helpers.
'''
	This script takes a simple 27-day persisted sw speed, and computed the cross correlation, chisquare, reduced chisquare.
	There is no sense of history and delay here since we explicitly look at 27 day persistence (the days have been defined). 

'''

from helperfn.utils import *

np.random.seed(0)

isize = 128
osize = 40
n_channel = 1
channelNo = 4
n_in = isize*isize*n_channel
learning_rate = 0.0001 #Put 3 0
kp=0.55
n_iter2 =120
kvalue = 1
n_out = 1
param = 0
#param_stddev = param+3
cv=int(sys.argv[1])
probab = 0.5
N_times = 1
N_models = 1
ch_filter = 193
# Let us define some useful functions
def Loader(paths):
	for path in paths:
		dat=np.load(path,allow_pickle=True).tolist()
		x_tmp=dat['input'][:,0].reshape([-1,1])
		y_tmp=dat['output'][:,0].reshape([-1,1])
		y_st_tmp=dat['output'][:,1].reshape([-1,1])
		try:
			x=np.concatenate([x,x_tmp],axis=0)
			y=np.concatenate([y,y_tmp],axis=0)
			y_st=np.concatenate([y_st,y_st_tmp],axis=0)
		except:
			x=x_tmp
			y=y_tmp
			y_st=y_st_tmp
	return x,y,y_st

path = 'Models/Persistence/'+str(ch_filter)+'/CV_'+str(cv)+'/27DayPersist/'
#if not os.path.isdir(path):
	#os.makedirs(path)
#--------------------------------------------------->
trainpaths = sorted(glob('CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Train/*.npy'))
testpaths = sorted(glob('CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))
validationpaths = sorted(glob('CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))

#_,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,[256,256,1],False)
#_,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,[256,256,1],False)
#yval = ytest
#train_dat=np.load(trainpaths,allow_pickle=True).tolist()
#xtrain=train_dat['input'][:,0]
#ytrain=train_dat['output'][:,0]
#ytrain_stddev=train_dat['output'][:,1]
#test_dat=np.load(testpaths,allow_pickle=True).tolist()
#xtest=test_dat['input'][:,0]
#ytest=test_dat['output'][:,0]
#ytest_stddev=test_dat['output'][:,1]
xtrain,ytrain,ytrain_stddev=Loader(trainpaths)
print ytrain.shape
xtest,ytest,ytest_stddev=Loader(testpaths)
xval = xtest
yval = ytest
#-------------------------------------------->
print xtest.shape 
print ytest.shape

correl=corr(xtest.ravel(),ytest.ravel())
mse=np.mean(np.square(xtest-ytest))
redmse=chi2(xtest,ytest,np.square(ytest_stddev))
test={'correl':correl,'mse':mse,'redmse':redmse}
print test 
print np.save(path+'Prediction.npy',[xtest,ytest])
# np.save(path+'Test_stats.npy',test)
# correl=corr(xtrain.ravel(),ytrain.ravel())
# mse=np.mean(np.square(xtrain-ytrain))
# redmse=chi2(xtrain,ytrain,np.square(ytrain_stddev))
# train={'correl':correl,'mse':mse,'redmse':redmse}
# np.save(path+'Train_stats.npy',train)