import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import sys
import os
from sklearn.externals import joblib
sys.path.append('../')
# Custom helpers.
from helperfn.utils import *
#-------

import xgboost as XGB

np.random.seed(0)

n_out = 1
param = 11
param_stddev = param+3
delay = int(sys.argv[2])
history = int(sys.argv[1])
cv=int(sys.argv[3])
probab = 0.5
N_times = 1
N_models = 1

ch_filter = 193

"""
	This piece of code regresses SW data with itself, using different histories and delays.
	The algorithm used here is Gradient Boosting using the XGBoost library.
"""
#Path to save the model
path = '../Models/XGBoost_SW/'+str(ch_filter)+'/CV_'+str(cv)+'/XGBoost_SW'+str(history)+str(delay)+'/'
if not os.path.isdir(path):
	os.makedirs(path)
#--------------------------------------------------->
#Data loading
trainpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Train/*.npy'))
testpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))
validationpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))

_,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,[256,256,1],False)
_,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,[256,256,1],False)
yval = ytest
#-------------------------------------------->
#Max-min normlization for data.
ymax =np.max(ytrain)-np.min(ytrain)
ymin =np.min(ytrain)

ytrain = NormalizeImage(ytrain,ymin,ymax)
ytest = NormalizeImage(ytest,ymin,ymax)
yval =NormalizeImage(yval,ymin,ymax)
#-------------------------------------------->
#We have our dataset ready. Let us get the training data proper.

'''
	Our dataset is of the form [global_batches,sequential_batch_size,size]. We need to take the inputs corresponding
	to take a window of length *history*, and map it to an observation at *history+delay*. Hen_ine, we need to keep
	moving this window, and derive a new dataset from the existing data for an easy mapping. The same has been
	explained in Technicalities.md
'''
xo=[]
yo=[]
yo2 =[]
for i in xrange(ytest.shape[0]):
	for j in xrange(ytest.shape[1]-delay-history):
		xo.append(ytest[i,j:j+history,:])
		yo.append(ytest[i,j+history+delay-1,:])
		yo2.append(ytest_stddev[i,j+history+delay-1,:])
xtest = np.reshape(np.asarray(xo),[-1,(history)*n_out])
ytest = np.reshape(np.asarray(yo),[-1,n_out])
ytest_stddev = np.reshape(np.asarray(yo2),[-1,n_out])

xo=[]
yo=[]
yo2=[]
for i in xrange(ytrain.shape[0]):
	for k in xrange(ytrain.shape[1]-delay-history):
		xo.append(ytrain[i,k:k+history,:])
		yo.append(ytrain[i,k+history+delay-1,:])
		yo2.append(ytrain_stddev[i,k+history+delay-1,:])
xtrain_condensed = np.reshape(np.asarray(xo),[-1,(history)*n_out])
ytrain_condensed = np.asarray(yo)
ytrain_condensed_stddev = np.asarray(yo2)

xval = xtest
yval = ytest
#-------------------------------------------->
#Define the xgboosting model. We use Grid Search to obtain the optimum parameters.
minerr = 0.0
TrainMat = XGB.DMatrix(xtrain_condensed,ytrain_condensed)
for et in [0.001,0.01,0.1,0.8,0.9,1.0]:
	for lamb in [50,10,5,1,0.5,0.05]:
		parameters = {'eta':et, 'seed':0, 'objective':'reg:linear','max_depth':200,'lambda':lamb}
		TrainedModel = XGB.train(parameters,TrainMat)
		TestMat = XGB.DMatrix(xtest)
		Prediction = np.reshape(TrainedModel.predict(TestMat),[-1,n_out])
		Prediction2 = np.reshape(TrainedModel.predict(XGB.DMatrix(xtrain_condensed)),[-1,n_out])
		print Prediction.shape
		print ytest.shape

		Prediction = SavingTestData(Prediction,ytest,ymin,ymax,ytest_stddev,path,'XGBoost')
		Prediction2 = SavingTrainingData(Prediction2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'XGBoost')
		corr_pred=[]
		corr_train=[]
		TrainE2=[]
		ValE2=[]

		for i in xrange(n_out):
			corr_pred.append(corr(Prediction[:,i],Prediction[:,i+n_out]))
			corr_train.append(corr(Prediction2[:,i],Prediction2[:,i+n_out]))
			TrainE2.append(np.mean(np.square(Prediction2[:,i]-Prediction2[:,i+n_out])))
			ValE2.append(np.mean(np.square(Prediction[:,i]-Prediction[:,i+n_out])))

		corr_pred = np.asarray(corr_pred)
		corr_train = np.asarray(corr_train)
		ValE2 = np.asarray(ValE2)
		TrainE2 = np.asarray(TrainE2)
		if et==0.001 and lamb==50:
			minerr = ValE2[-1]

		if ValE2[-1]<=minerr:
			print "------------------"
			pars = parameters
			minerr = ValE2[-1]
TrainedModel = XGB.train(pars,TrainMat)
TestMat = XGB.DMatrix(xtest)
Prediction = np.reshape(TrainedModel.predict(TestMat),[-1,n_out])
Prediction2 = np.reshape(TrainedModel.predict(XGB.DMatrix(xtrain_condensed)),[-1,n_out])
print Prediction.shape
print ytest.shape
#-------------------------------------------->
#Saving the data.
Prediction = SavingTestData(Prediction,ytest,ymin,ymax,ytest_stddev,path,'XGBoost')
Prediction2 = SavingTrainingData(Prediction2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'XGBoost')
joblib.dump(TrainedModel,path+'XGBoost_'+str(history)+str(delay)+'.pkl')
#Use joblib to load and use the model.
Prediction = NormalizeImage(Prediction,ymin,ymax)
ytest_stddev = NormalizeImage(ytest_stddev,0.0,ymax)
#-------------------------------------------->
#Saving the correlations.
for i in xrange(n_out):
	try:
		corr_pred.append(corr(Prediction[:,i],Prediction[:,i+n_out]))
		corr_train.append(corr(Prediction2[:,i],Prediction2[:,i+n_out]))
		TrainE2.append(np.mean(np.square(Prediction2[:,i]-Prediction2[:,i+n_out])))
		ValE2.append(np.mean(np.square(Prediction[:,i]-Prediction[:,i+n_out])))
	except:
		corr_pred = [corr(Prediction[:,i],Prediction[:,i+n_out])]
		corr_train = [corr(Prediction2[:,i],Prediction2[:,i+n_out])]
		TrainE2 = [np.mean(np.square(Prediction2[:,i]-Prediction2[:,i+n_out]))]
		ValE2 = [np.mean(np.square(Prediction[:,i]-Prediction[:,i+n_out]))]
corr_pred = np.asarray(corr_pred)
corr_train = np.asarray(corr_train)
ValE2 = np.asarray(ValE2)
TrainE2 = np.asarray(TrainE2)
#-------------------------------------------->
print "Mean Prediction correlation: " + str(np.mean(corr_pred))
print "Mean training correlation: " + str(np.mean(corr_train))
corel = corr(Prediction[:,i],Prediction[:,i+n_out],'full')
print "Maximum is at: " + str(np.where(corel==np.max(corel)))
print "Maximum must be at: " + str(corel.shape[0]/2)

print Prediction.shape
print ytest_stddev.shape

#-------------------------------------------->
print "Chi2 reduced error of testing data is ",
print chi2(Prediction[:,0],Prediction[:,1],ytest_stddev)
print "Testing MSError is ",
print ValE2
np.savetxt(path+'AE_Predict_TrainError'+str(history)+str(delay)+'.txt',TrainE2)
np.savetxt(path+'AE_Predict_ValidError'+str(history)+str(delay)+'.txt',ValE2 )
np.savetxt(path+'AE_Predict_TrainCorr'+str(history)+str(delay)+'.txt',corr_train)
np.savetxt(path+'AE_Predict_ValidCorr'+str(history)+str(delay)+'.txt',corr_pred )
#-------------------------------------------->
'''
	If you would want to plot the results and see, uncomment the plot commands below.
'''
# fig = plt.figure(3)
# t=xrange(Prediction.shape[0])
# plt.plot(t,Prediction[:,0],'r',label = 'Prediction',linewidth = 1.0)
# plt.plot(t,Prediction[:,0],'ro',markersize = 2.0)
# plt.plot(t,Prediction[:,n_out],'g',label = 'Given data',linewidth = 1.0)
# plt.plot(t,Prediction[:,n_out],'go',markersize = 2.0)
# plt.legend(loc = 'upper right')
#
# fig,ax = plt.subplots()
# ax.scatter(Prediction[:,0],Prediction[:,n_out])
# lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()])]
# lims2 = [np.max([ax.get_xlim(), ax.get_ylim()]),np.min([ax.get_xlim(), ax.get_ylim()])]
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.plot(lims,lims,'k')
# ax.plot(lims,lims2,'k')
# ax.set_xlabel('Prediction')
# ax.set_ylabel('Observation')
# plt.show()
