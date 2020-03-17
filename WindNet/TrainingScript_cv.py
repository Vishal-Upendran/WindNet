import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import shutil
import sys
import os
sys.path.append('../')
#-----------------
#211 channel or 193 channel from AIA on SDO? Needed to import the preliminary variables. Change this channel no in Preliminary_vars.
#---------------
#Custom helpers
from helperfn.utils import *
from helperfn.tf_utils import *
from WindNet import WindNet
from helperfn.Preliminary_vars import *
#Preliminary_vars imports the path to data, data shape, etc.
#----
'''
    This script builds the WindNet model and trains it. All the constants are explained in the Technicalities.md file.
'''
#isize = 224
#n_channel = 3

#isize_before_pretrained = 128
#n_channel_before_pretrained = 1 #Was 8

#channelNo = 4 #Was 4

#n_in = isize*isize*n_channel

#inshape=[isize_before_pretrained,isize_before_pretrained,8]
#n_in_before_pretrained = isize_before_pretrained*isize_before_pretrained*n_channel_before_pretrained
learning_rate = 0.0005 #Was 0.00001 for others
kp=0.55
kvalue = 1
isize = 224
n_channel = 3

isize_before_pretrained = 256
n_channel_before_pretrained = 1

#------------
#ch_filter is either 193 or 211. It must be defined before importing this module.
ch_filter = 211
#------------
channelNo = 0
#This channelNo is the index number in the channel of data. For the individual 211 and 193 data, we have this as 0.

n_in = isize*isize*n_channel
n_in_before_pretrained = isize_before_pretrained*isize_before_pretrained*n_channel_before_pretrained
inshape=[isize_before_pretrained,isize_before_pretrained,n_channel_before_pretrained]
n_out = 1
param = 11 #Was 11
param_stddev = param+3 #Was param+3
nc = 832

#n_out = 1
#param = 11 #Was 11
#param_stddev = param+3 #was +2
delay = int(sys.argv[1])
history = int(sys.argv[2])
n_iter2 = int(sys.argv[3]) #200
cross_valid = int(sys.argv[4])
probab = 0.5
Direc='/tmp/tenorflow/run1'
#nc = 832

print delay
print history
print n_iter2

np.random.seed(0)
tf.set_random_seed(0)
#tv1=1000
#tv2=20000
#Path for saving the model.
path = '../Models/WindNet/CrossValidation/'+str(ch_filter)+'/CV_'+str(cross_valid)+'/SDOPred'+str(history)+str(delay)+'/'
if os.path.exists(path+'AE_Predict_ValidChi2'+str(history)+str(delay)+'.txt'):
	sys.exit(0)

if not os.path.isdir(path):
	os.makedirs(path)
else:
    shutil.rmtree(path)
    os.makedirs(path)
path_for_model = path+'Model_save/'
if not os.path.isdir(path_for_model):
    os.makedirs(path_for_model)
#------------------------------------------------------>

print "Data loading started"
trainpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_%d/Train/*.npy'%cross_valid))
testpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_%d/Test/*.npy'%cross_valid))
validationpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_%d/Test/*.npy'%cross_valid))


xtrain,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,inshape,True,channelNo)
xtest,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,inshape,True,channelNo)
xval,yval,yval_stddev = BifurcatedDataloader(validationpaths,param,param_stddev,inshape,True,channelNo)
print xtrain.shape
print ytrain.shape
print xtest.shape
print ytest.shape
print "Data loading: Done"
#------------------------------------------------------>
#Data normlization.
xmean = np.mean(xtrain)
xstd = 1.0# np.std(xtrain)
xtrain = NormalizeImage(xtrain,xmean,xstd)
xtest = NormalizeImage(xtest,xmean,xstd)
xval = NormalizeImage(xval,xmean,xstd)

ymax =np.max(ytrain,axis = (0,1))-np.min(ytrain,axis = (0,1))
ymin =np.min(ytrain,axis = (0,1))
ytrain = NormalizeImage(ytrain,ymin,ymax)
ytest = NormalizeImage(ytest,ymin,ymax)
yval =NormalizeImage(yval,ymin,ymax)

ytrain_stddev = NormalizeImage(ytrain_stddev,0.0,ymax)
ytest_stddev = NormalizeImage(ytest_stddev,0.0,ymax)
yval_stddev = NormalizeImage(yval_stddev,0.0,ymax)

print xtrain.shape
print xtest.shape
print xval.shape
print ytrain.shape
print ytest.shape
print yval.shape
print "Data Normalization: Done"
#------------------------------------------------------>
#Data windowing. Explained in Technicalities.md
xtest,ytest,ytest_stddev = DataWindowParser(xtest,ytest,ytest_stddev,history,delay)
xval,yval,yval_stddev = DataWindowParser(xval,yval,yval_stddev,history,delay)
xtrain = np.reshape(xtrain,[-1,20,isize,isize,n_channel])
xtest = np.reshape(xtest,[-1,isize,isize,n_channel])
xval = np.reshape(xval,[-1,isize,isize,n_channel])
print "Data preprocessing: Done"
#------------------------------------------------------>
print "Model definition"
tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None,isize,isize,n_channel])
y = tf.placeholder(tf.float32,[None,n_out])
y_stddev = tf.placeholder(tf.float32,[None,n_out])
y2 = tf.placeholder(tf.float32,[None,history,n_out])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
xprime = tf.placeholder(tf.float32,[None,history,nc])
sess = tf.Session()
hidden_units = 400

#it = np.load('IterationList.npy').tolist()
#it = it[str(history)+str(delay)]
#path_to_model = '/home/vishal_u/WindNet/ForPlotting/WindNet/SDOPred'+str(history)+str(delay)+'/Prev_save/model_no'+str(it)
'''Feed in an appropriate path to the WindNet class if you want to start pretrained.
'''
#Make new instance of the model
WindModel = WindNet(history,delay,lr,x,y,y_stddev,y2,keep_prob,xprime,sess,hidden_units)
#Build the network.
WindModel.BuildNetwork()
#Normalize the embeddings.
WindModel.EmbeddingNormalization(xtrain)
print "Model definition: Done"
#------------------------------------------------------>
#Get Embeddings.
FCtrain = []
for i in xrange(xtrain.shape[0]):
    FCtrain.append(sess.run(WindModel.FClayer,feed_dict = {x:xtrain[i,:,:,:,:]}))
FCtrain = np.asarray(FCtrain)
FCtest = np.reshape(sess.run(WindModel.FClayer,feed_dict = {x:xtest}),[-1,history,WindModel.endshp])

#Data window paser.
xtrain_condensed,ytrain_condensed_stddev,ytrain_condensed = DataWindowParser(FCtrain,ytrain_stddev,ytrain,history,delay)

xo2=[]
for i in xrange(FCtrain.shape[0]):
    for k in xrange(FCtrain.shape[1]-delay-history):
        xo2.append(xtrain[i,k:k+history,:])
xtrain_historied = np.reshape(np.asarray(xo2),[-1,history,isize,isize,n_channel])
xtrain_condensed = np.reshape(xtrain_condensed,[-1,history,nc])

xtrain_noisy = FCtrain#+np.random.normal(0.0,0.1,[FCtrain.shape[0],FCtrain.shape[1],FCtrain.shape[2]])
#xtrain_noisy= np.vstack((xtrain_noisy,FCtrain))
#ytrain  = np.vstack((ytrain,ytrain))
#ytrain_stddev = np.vstack((ytrain_stddev,ytrain_stddev))
print "Training data reshaped!"
#------------------------------------------------------>
#saver=tf.train.Saver(tf.all_variables(),max_to_keep = -1)
WindModel.saver_init()

TrainE2=[]
ValE2=[]
TestingCorr=[]
TrainingCorr=[]
Testingchi2=[]
#Train the model.
for i in xrange(n_iter2):
    tr = 0
    tr2 = 0
    ptr=1
    for p in xrange(1):
        rng = np.arange(xtrain.shape[0])
        #np.random.shuffle(rng)
        for j in rng:
            xo=[]
            yo=[]
            yo2=[]
            yo3=[]
            for k in xrange(xtrain.shape[1]-delay-history):
                xo.append(xtrain_noisy[j,k:k+history,:])
                yo.append(ytrain[j,k+history+delay-1,:])
                yo2.append(ytrain[j,k+delay-1:k+history+delay-1,:])
                yo3.append(ytrain_stddev[j,k+history+delay-1,:])
            xo = np.reshape(np.asarray(xo),[-1,history,nc])
            yo = np.reshape(np.asarray(yo),[-1,n_out])
            yo2 = np.reshape(np.asarray(yo2),[-1,history,n_out])
            yo3 = np.reshape(np.asarray(yo3),[-1,n_out])
            _,tr = sess.run([WindModel.optimizer_pred,WindModel.cost], feed_dict = {y_stddev:yo3,keep_prob: 0.3,xprime:xo, y:yo,y2:yo2, lr: learning_rate*1.0}) #*np.exp(-2.303*i/n_iter2)
            tr2 = tr2+tr
            ptr=ptr+1

    TrainE2.append(tr2/ptr)
    tr=0
    tr2 = 0
    k=1
    tr = sess.run(WindModel.cost,feed_dict = {y_stddev:ytest_stddev,keep_prob: 1.0,xprime:FCtest,y: ytest})
    tr3 = sess.run(WindModel.chi2_reduced,feed_dict = {y_stddev:ytest_stddev,keep_prob: 1.0,xprime:FCtest,y: ytest})
    tv = sess.run(WindModel.Regression,feed_dict = {xprime: FCtest, y: ytest, keep_prob:1.0})
    value = SavingTestData(tv,ytest,ymin,ymax,ytest_stddev,path,'WindNet',i)
    TestingCorr.append(corr(value[:,0],value[:,1]))
    ValE2.append(tr)
    Testingchi2.append(tr3)
    if TestingCorr[-1]>=0.5:
        WindModel.save_model(path_for_model+'model_no-'+str(i))
        tv2 = sess.run(WindModel.Regression, feed_dict = {xprime: xtrain_condensed,keep_prob:1.0 })
        value= SavingTrainingData(tv2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'WindNet',i)
    else:
        pass 

    if i%10==0:
        WindModel.save_model(path_for_model+'model_no-'+str(i))
        #print sp
        print "Iteration: " +str(i+1)
        print "Training error is " +str(TrainE2[i]),
        print ", and Validation error is " + str(ValE2[i])
        print "Testing corr: "+ str(TestingCorr[i])
        print "---------"
        tv2 = sess.run(WindModel.Regression, feed_dict = {xprime: xtrain_condensed,keep_prob:1.0 })
        value= SavingTrainingData(tv2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'WindNet',i)
        TrainingCorr.append(corr(value[:,0],value[:,1]))
print "Training done!"
#------------------------------------------------------>
sp = WindModel.save_model(path_for_model+'model_no-'+str(i))
tv = sess.run(WindModel.Regression,feed_dict = {xprime: FCtest, y: ytest, keep_prob:1.0})
Prediction = SavingTestData(tv,ytest,ymin,ymax,ytest_stddev,path,'WindNet',i)
tv2 = sess.run(WindModel.Regression, feed_dict = {xprime: xtrain_condensed,keep_prob:1.0 })
Prediction2 = SavingTrainingData(tv2,ytrain_condensed,ymin,ymax,ytrain_condensed_stddev,path,'WindNet',i)
#Predictions for training and testing sets done.
#------------------------------------------------------>
#Correlation plotting time!
Prediction=NormalizeImage(Prediction,ymin,ymax)
Prediction2=NormalizeImage(Prediction2,ymin,ymax)

for i in xrange(n_out):
	try:
		corr_pred.append(corr(Prediction[:,i],Prediction[:,i+n_out]))
		corr_train.append(corr(Prediction2[:,i],Prediction2[:,i+n_out]))
	except:
		corr_pred = [corr(Prediction[:,i],Prediction[:,i+n_out])]
		corr_train = [corr(Prediction2[:,i],Prediction2[:,i+n_out])]
corr_pred = np.asarray(corr_pred)
corr_train = np.asarray(corr_train)

print "Mean Prediction correlation: " + str(np.mean(corr_pred))
print "Mean training correlation: " + str(np.mean(corr_train))
print "Min Validation error location " + str(np.where(ValE2==np.min(ValE2)))

np.savetxt(path+'AE_Predict_TrainError'+str(history)+str(delay)+'.txt',TrainE2)
np.savetxt(path+'AE_Predict_ValidError'+str(history)+str(delay)+'.txt',ValE2 )
np.savetxt(path+'AE_Predict_TrainCorr'+str(history)+str(delay)+'.txt',TrainingCorr)
np.savetxt(path+'AE_Predict_ValidCorr'+str(history)+str(delay)+'.txt',TestingCorr )
np.savetxt(path+'AE_Predict_ValidChi2'+str(history)+str(delay)+'.txt',Testingchi2 )

#Plotting. Uncomment the code below to plot after training.

plt.figure(1)
t=np.arange(n_iter2)
plt.plot(t,np.log10(TrainE2),'r',label = "TrainError")
plt.plot(t,np.log10(ValE2),'g',label = 'ValError')
plt.legend(loc = 'upper right')

fig = plt.figure(3)
t=xrange(Prediction.shape[0])
plt.plot(t,Prediction[:,0],'r',label = 'Prediction',linewidth = 1.0)
plt.plot(t,Prediction[:,0],'ro')
plt.plot(t,Prediction[:,n_out],'g',label = 'Given data',linewidth = 1.0)
#plt.plot(t,Prediction[:,n_out],'go')
plt.errorbar(t,Prediction[:,n_out],yerr = ytest_stddev,fmt = 'go')
plt.legend(loc = 'upper right')

fig,ax = plt.subplots()
ax.scatter(Prediction[:,0],Prediction[:,n_out])
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()])]
lims2 = [np.max([ax.get_xlim(), ax.get_ylim()]),np.min([ax.get_xlim(), ax.get_ylim()])]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.plot(lims,lims,'k')
ax.plot(lims,lims2,'k')
ax.set_xlabel('Prediction')
ax.set_ylabel('Observation')

plt.show()
