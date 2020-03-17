import numpy as np
import os
from glob import glob
trainpaths = sorted(glob('../Bifurcated_data/CleanData/Train/*.npy'))
testpaths = sorted(glob('../Bifurcated_data/CleanData/Test/*.npy'))
validationpaths = sorted(glob('../Bifurcated_data/CleanData/Test/*.npy'))

_,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,False)
_,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,False)

ymax =np.max(ytrain)-np.min(ytrain)
ymin =np.min(ytrain)

ytrain = NormalizeImage(ytrain,ymin,ymax)
ytest = NormalizeImage(ytest,ymin,ymax)

xo=[]
yo=[]
yo2 =[]
for i in xrange(ytest.shape[0]):
	for j in xrange(ytest.shape[1]-delay-history):
		xo.append(ytest[i,j:j+history,:])
		yo.append(ytest[i,j+history+delay,:])
		yo2.append(ytest_stddev[i,j+history+delay,:])
xtest = np.reshape(np.asarray(xo),[-1,(history)*n_out])
ytest = np.reshape(np.asarray(yo),[-1,n_out])
ytest_stddev = np.reshape(np.asarray(yo2),[-1,n_out])

xo=[]
yo=[]
yo2=[]
for i in xrange(ytrain.shape[0]):
	for k in xrange(ytrain.shape[1]-delay-history):
		xo.append(ytrain[i,k:k+history,:])
		yo.append(ytrain[i,k+history+delay,:])
		yo2.append(ytrain_stddev[i,k+history+delay,:])
xtrain_condensed = np.reshape(np.asarray(xo),[-1,(history)*n_out])
ytrain_condensed = np.asarray(yo)
ytrain_condensed_stddev = np.asarray(yo2)
