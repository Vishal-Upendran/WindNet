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
#----

np.random.seed(0)

n_out = 1
param = 11
param_stddev = param+3
delay = int(sys.argv[2])
history = int(sys.argv[1])
cv=int(sys.argv[3])
probab = 0.5
ch_filter = 193
"""
	This code regresses Solar wind data against itself with different histories and delays.
	Support vector regression is done using Radial Basis Functions, Polynomial and Linear regression fits.
"""

#Path to save the model.
path = '../Models/SVM_SW/'+str(ch_filter)+'/CV_'+str(cv)+'/SVM_SW'+str(history)+str(delay)+'/'
if not os.path.isdir(path):
	os.makedirs(path)
#path for saving the model and outpus.
#--------------------------------------------------->
#Data loading.
trainpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Train/*.npy'))
testpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))
validationpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/*.npy'))

_,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,[256,256,1],False)
_,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,[256,256,1],False)
yval = ytest
print "Data loading: done"
#--------------------------------------------------->
#Max-min data normlization.
ymax =np.max(ytrain)-np.min(ytrain)
ymin =np.min(ytrain)

ytrain = NormalizeImage(ytrain,ymin,ymax)
ytest = NormalizeImage(ytest,ymin,ymax)
yval =NormalizeImage(yval,ymin,ymax)
print "Data preprocessing: done"
#--------------------------------------------------->
'''
	Our dataset is of the form [global_batches,sequential_batch_size,size]. We need to take the inputs corresponding
	to take a window of length *history*, and map it to an observation at *history+delay*. Hence, we need to keep
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
#--------------------------------------------------->
xval = xtest
yval = ytest
print "Data loading done"
#Define the SVM kernels and regressors.
RBFModel = Regressor(kernel = 'rbf', C=1e4, gamma = 0.001)
LinearModel = Regressor(kernel = 'linear', C=1e+4)
PolynomialModel = Regressor(kernel = 'poly', C=1e+4 ,degree = 5)

RBFModel.fit(xtrain_condensed,ytrain_condensed.ravel())
PolynomialModel.fit(xtrain_condensed,ytrain_condensed.ravel())
LinearModel.fit(xtrain_condensed,ytrain_condensed.ravel())
#That was technically training.

#--------------------------------------------------->
#Testing phase.
RBFPrediction = np.reshape(RBFModel.predict(xtest),[-1,n_out])
PolyPrediction = np.reshape(PolynomialModel.predict(xtest),[-1,n_out])
LinearPrediction =np.reshape(LinearModel.predict(xtest),[-1,n_out])

RBFPrediction2 = np.reshape(RBFModel.predict(xtrain_condensed),[-1,n_out])
PolyPrediction2 = np.reshape(PolynomialModel.predict(xtrain_condensed),[-1,n_out])
LinearPrediction2 =np.reshape(LinearModel.predict(xtrain_condensed),[-1,n_out])

print RBFPrediction.shape
print PolyPrediction.shape
print LinearPrediction.shape
print ytest.shape
#--------------------------------------------------->
#Saving the data,
RBFPrediction = SavingTestData(RBFPrediction,ytest,ymin,ymax,ytest_stddev,path,'RBF')
PolyPrediction = SavingTestData(PolyPrediction,ytest,ymin,ymax,ytest_stddev,path,'Poly')
LinearPrediction = SavingTestData(LinearPrediction,ytest,ymin,ymax,ytest_stddev,path,'Linear')

RBFPrediction2 = SavingTrainingData(RBFPrediction2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'RBF')
PolyPrediction2 = SavingTrainingData(PolyPrediction2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'Poly')
LinearPrediction2 = SavingTrainingData(LinearPrediction2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'Linear')

joblib.dump(RBFModel,path+'SWSVM_RBF_'+str(history)+str(delay)+'.pkl')
joblib.dump(PolynomialModel,path+'SWSVM_Poly_'+str(history)+str(delay)+'.pkl')
joblib.dump(LinearModel,path+'SWSVM_Linear_'+str(history)+str(delay)+'.pkl')
#--------------------------------------------------->
#Saving the correlation.
for i in xrange(n_out):
	try:
		corr_predRBF.append(corr(RBFPrediction[:,i],RBFPrediction[:,i+n_out]))
		corr_predPoly.append(corr(PolyPrediction[:,i],PolyPrediction[:,i+n_out]))
		corr_predLinear.append(corr(LinearPrediction[:,i],LinearPrediction[:,i+n_out]))

		corr_trainRBF.append(corr(RBFPrediction2[:,i],RBFPrediction2[:,i+n_out]))
		corr_trainPoly.append(corr(PolyPrediction2[:,i],PolyPrediction2[:,i+n_out]))
		corr_trainLinear.append(corr(LinearPrediction2[:,i],LinearPrediction2[:,i+n_out]))

		RBFTrainE2.append(np.mean(np.square(RBFPrediction2[:,i]-RBFPrediction2[:,i+n_out])))
		RBFValE2.append(np.mean(np.square(RBFPrediction[:,i]-RBFPrediction[:,i+n_out])))
		LinearTrainE2.append(np.mean(np.square(LinearPrediction2[:,i]-LinearPrediction2[:,i+n_out])))
		LinearValE2.append(np.mean(np.square(LinearPrediction[:,i]-LinearPrediction[:,i+n_out])))
		PolyTrainE2.append(np.mean(np.square(PolyPrediction2[:,i]-PolyPrediction2[:,i+n_out])))
		PolyValE2.append(np.mean(np.square(PolyPrediction[:,i]-PolyPrediction[:,i+n_out])))
	except:
		corr_predRBF = [(corr(RBFPrediction[:,i],RBFPrediction[:,i+n_out]))]
		corr_predPoly = [(corr(PolyPrediction[:,i],PolyPrediction[:,i+n_out]))]
		corr_predLinear= [(corr(LinearPrediction[:,i],LinearPrediction[:,i+n_out]))]

		corr_trainRBF = [(corr(RBFPrediction2[:,i],RBFPrediction2[:,i+n_out]))]
		corr_trainPoly = [(corr(PolyPrediction2[:,i],PolyPrediction2[:,i+n_out]))]
		corr_trainLinear = [(corr(LinearPrediction2[:,i],LinearPrediction2[:,i+n_out]))]

		RBFTrainE2 = [(np.mean(np.square(RBFPrediction2[:,i]-RBFPrediction2[:,i+n_out])))]
		RBFValE2 = [(np.mean(np.square(RBFPrediction[:,i]-RBFPrediction[:,i+n_out])))]
		LinearTrainE2= [(np.mean(np.square(LinearPrediction2[:,i]-LinearPrediction2[:,i+n_out])))]
		LinearValE2= [(np.mean(np.square(LinearPrediction[:,i]-LinearPrediction[:,i+n_out])))]
		PolyTrainE2= [(np.mean(np.square(PolyPrediction2[:,i]-PolyPrediction2[:,i+n_out])))]
		PolyValE2 = [(np.mean(np.square(PolyPrediction[:,i]-PolyPrediction[:,i+n_out])))]

corr_predLinear = np.asarray(corr_predLinear)
corr_predRBF = np.asarray(corr_predRBF)
corr_predPoly = np.asarray(corr_predPoly)

corr_trainLinear = np.asarray(corr_trainLinear)
corr_trainRBF = np.asarray(corr_trainRBF)
corr_trainPoly = np.asarray(corr_trainPoly)

RBFTrainE2 = np.asarray(RBFTrainE2)
RBFValE2 = np.asarray(RBFValE2)
LinearTrainE2 = np.asarray(LinearTrainE2)
LinearValE2 = np.asarray(LinearValE2)
PolyTrainE2 = np.asarray(PolyTrainE2)
PolyValE2 = np.asarray(PolyValE2)
#--------------------------------------------------->
print "Mean Prediction correlation: Radial Basis Kernel " + str(np.mean(corr_predRBF))
print "Mean Prediction correlation: Linear Kernel " + str(np.mean(corr_predLinear))
print "Mean Prediction correlation: Polynomial Kernel " + str(np.mean(corr_predPoly))

print "Mean training correlation: Radial Basis Kernel " + str(np.mean(corr_trainRBF))
print "Mean training correlation: Linear Kernel " + str(np.mean(corr_trainLinear))
print "Mean training correlation: Polynomial Kernel " + str(np.mean(corr_trainPoly))

corel = corr(RBFPrediction[:,0],ytest[:,0],'full')
print "Maximum for RBF is at: " + str(np.where(corel==np.max(corel)))
corel = corr(LinearPrediction[:,0],ytest[:,0],'full')
print "Maximum for Linear is at: " + str(np.where(corel==np.max(corel)))
corel = corr(PolyPrediction[:,0],ytest[:,0],'full')
print "Maximum for Polynomial is at: " + str(np.where(corel==np.max(corel)))
print "Maximum must be at: " + str(corel.shape[0]/2)
#--------------------------------------------------->
np.savetxt(path+'RBF_TrainError'+str(history)+str(delay)+'.txt',RBFTrainE2)
np.savetxt(path+'RBF_ValidError'+str(history)+str(delay)+'.txt',RBFValE2 )
np.savetxt(path+'Poly_TrainError'+str(history)+str(delay)+'.txt',PolyTrainE2)
np.savetxt(path+'Poly_ValidError'+str(history)+str(delay)+'.txt',PolyValE2 )
np.savetxt(path+'Linear_TrainError'+str(history)+str(delay)+'.txt',LinearTrainE2)
np.savetxt(path+'Linear_ValidError'+str(history)+str(delay)+'.txt',LinearValE2 )

np.savetxt(path+'RBF_Predict'+str(history)+str(delay)+'.txt',corr_predRBF)
np.savetxt(path+'Linear_Predict'+str(history)+str(delay)+'.txt',corr_predLinear)
np.savetxt(path+'Poly_Predict'+str(history)+str(delay)+'.txt',corr_predPoly)
np.savetxt(path+'RBF_Train'+str(history)+str(delay)+'.txt',corr_trainRBF)
np.savetxt(path+'Linear_Train'+str(history)+str(delay)+'.txt',corr_trainLinear)
np.savetxt(path+'Poly_Train'+str(history)+str(delay)+'.txt',corr_trainPoly)
#--------------------------------------------------->
'''
	If you would want to plot the results and see, uncomment the plot commands below.
'''

# fig = plt.figure(3)
# plt.subplot(3,1,1)
# t=xrange(RBFPrediction.shape[0])
# plt.plot(t,RBFPrediction[:,0],'r',label = 'Prediction',linewidth = 1.0)
# plt.plot(t,RBFPrediction[:,0],'ro',markersize = 3)
# plt.plot(t,ytest[:,0],'g',label = 'Given data',linewidth = 1.0)
# plt.plot(t,ytest[:,0],'go',markersize = 3)
# plt.title('RBF Prediction using Solar Wind alone')
# plt.legend(loc = 'upper right')
# plt.subplot(3,1,2)
# t=xrange(RBFPrediction.shape[0])
# plt.plot(t,LinearPrediction[:,0],'r',label = 'Prediction',linewidth = 1.0)
# plt.plot(t,LinearPrediction[:,0],'ro',markersize = 3)
# plt.plot(t,ytest[:,0],'g',label = 'Given data',linewidth = 1.0)
# plt.plot(t,ytest[:,0],'go',markersize = 3)
# plt.title('Linear SVM prediction using Solar Wind alone')
# plt.legend(loc = 'upper right')
# plt.subplot(3,1,3)
# t=xrange(RBFPrediction.shape[0])
# plt.plot(t,PolyPrediction[:,0],'r',label = 'Prediction',linewidth = 1.0)
# plt.plot(t,PolyPrediction[:,0],'ro',markersize = 3)
# plt.plot(t,ytest[:,0],'g',label = 'Given data',linewidth = 1.0)
# plt.plot(t,ytest[:,0],'go',markersize = 3)
# plt.title('Polynomial SVM prediction using Solar Wind alone')
# plt.legend(loc = 'upper right')
#
# fig,ax = plt.subplots()
# ax.scatter(RBFPrediction[:,0],ytest[:,0])
# lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()])]
# lims2 = [np.max([ax.get_xlim(), ax.get_ylim()]),np.min([ax.get_xlim(), ax.get_ylim()])]
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.plot(lims,lims,'k')
# ax.plot(lims,lims2,'k')
# ax.set_xlabel('Prediction')
# ax.set_ylabel('Observation')
# plt.show()
