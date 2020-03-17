import numpy as np 
import sys 
import os 
from glob import glob 
sys.path.append('../')
from helperfn.utils import *

ch_filter = 193
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
bp = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
history = int(sys.argv[1])
delay = int(sys.argv[2])
cv=int(sys.argv[3])

SavePath = '../Models/NaiveMean/'+str(ch_filter)+'/CV_'+str(cv)+'/Persist'+str(history)+str(delay)+'/'
if not os.path.isdir(SavePath):
	os.makedirs(SavePath)

iter_no=0

trainpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Train/*.npy'))
testpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))
validationpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))

_,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,[256,256,1],False)
_,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,[256,256,1],False)
ymin = np.min(ytrain)
ymax = np.max(ytrain)-ymin
xo=[]
yo=[]
yo2 =[]
for i in xrange(ytest.shape[0]):
	for j in xrange(ytest.shape[1]-delay-history):
		xo.append(ytest[i,j,:])
		yo.append(ytest[i,j+history+delay-1,:])
		yo2.append(ytest_stddev[i,j+history+delay-1,:])
xtest = np.reshape(np.asarray(xo),[-1])
ytest = np.reshape(np.asarray(yo),[-1])
ytest_stddev = np.reshape(np.asarray(yo2),[-1])

Error = ytest_stddev
Testfile = ytest

Mean_pred = np.mean(Testfile)*np.ones(Testfile.shape[0])
Red_MSE = chi2(Mean_pred,xtest,np.square(Error))
MSE = chi2(Mean_pred,xtest,Error*0.0+1.0)
correlation = corr(Mean_pred,xtest)

print MSE 
print Red_MSE
print correlation

val=SavingTestData(np.reshape(Mean_pred,[-1,1]),np.reshape(xtest,[-1,1]),0.0,1.0,Error,SavePath,'NaiveMean',0)

print val.shape
