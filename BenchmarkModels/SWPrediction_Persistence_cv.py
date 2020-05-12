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
param = 11
param_stddev = param+3
delay = int(sys.argv[2])
history = int(sys.argv[1])
cv=int(sys.argv[3])
probab = 0.5
N_times = 1
N_models = 1
ch_filter = 211
"""
	This code regresses Solar wind data against itself with different histories and delays.
	Support vector regression is done. We Radial Basis Functions, Polynomial and Linear regression fits for the same
"""

# Let us define some useful functions


path = '../Models/Persistence/'+str(ch_filter)+'/CV_'+str(cv)+'/Persist'+str(history)+str(delay)+'/'
if not os.path.isdir(path):
	os.makedirs(path)
#--------------------------------------------------->
trainpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Train/*.npy'))
testpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))
validationpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))

_,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,[256,256,1],False)
_,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,[256,256,1],False)
yval = ytest
#-------------------------------------------->
print ytest.shape
ymax =1.0 #np.max(ytrain)-np.min(ytrain)
ymin =0.0 #np.min(ytrain)

ytrain = NormalizeImage(ytrain,ymin,ymax)
ytest = NormalizeImage(ytest,ymin,ymax)
yval =NormalizeImage(yval,ymin,ymax)
#-------------------------------------------->
'''
	Our dataset is of the form [global_batches,sequential_batch_size,size]. We need to take the inputs corresponding
	to take a window of length *history*, and map it to an observation at *history+delay*. Hen_ine, we need to keep
	moving this window, and derive a new dataset from the existing data for an easy mapping. The same has been
	explained in the README associated with the utils file.
'''
xo=[]
yo=[]
yo2 =[]
for i in xrange(ytest.shape[0]):
	for j in xrange(ytest.shape[1]-delay-history+1):
		xo.append(ytest[i,j,:])
		yo.append(ytest[i,j+history+delay-1,:])
		yo2.append(ytest_stddev[i,j+history+delay-1,:])
xtest = np.reshape(np.asarray(xo),[-1,1,n_out])
ytest = np.reshape(np.asarray(yo),[-1,n_out])
ytest_stddev = np.reshape(np.asarray(yo2),[-1,n_out])

xo=[]
yo=[]
yo2=[]
for i in xrange(ytrain.shape[0]):
	for k in xrange(ytrain.shape[1]-delay-history+1):
		xo.append(ytrain[i,k,:])
		yo.append(ytrain[i,k+history+delay-1,:])
		yo2.append(ytrain_stddev[i,k+history+delay-1,:])
xtrain_condensed = np.reshape(np.asarray(xo),[-1,1,n_out])
ytrain_condensed = np.asarray(yo)
ytrain_condensed_stddev = np.asarray(yo2)
#-------------------------------------------->
xval = xtest
yval = ytest
print "Data loading done"

_=SavingTestData(xtest[:,-1,:],ytest,ymin,ymax,ytest_stddev,path,'Persist',0)
_=SavingTrainingData(xtrain_condensed[:,-1,:],ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'Persist',0)
