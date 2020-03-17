import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import zoom 
import matplotlib.pyplot as plt
import random
from glob import glob
import sys
import os
import argparse
sys.path.append('../')
#Custom helpers
from helperfn.utils import *
from helperfn.tf_utils import *
from WindNet import *
from helperfn.Preliminary_vars import *
#print ch_filter
#----
'''
    Define the constants first.
'''

learning_rate = 0.0001 #Was 0.00001 for others
kp=0.55
isize = 224
n_channel = 3
isize_before_pretrained = 256 #Was 128
n_channel_before_pretrained = 1

#------------
channelNo = 0 #Was 4
#This channelNo is the index number in the channel of data. For the individual 211 and 193 data, we have this as 0. 

n_in = isize*isize*n_channel
n_in_before_pretrained = isize_before_pretrained*isize_before_pretrained*n_channel_before_pretrained
inshape=[isize_before_pretrained,isize_before_pretrained,n_channel_before_pretrained]
n_out = 1
param = 11 #Was 11
param_stddev = param+3 #Was param+3
nc = 832
def ModelRestore(history,delay,n_iter2,path_for_model,ch_filter,cross_valid):

    #---------------------
    GCsavepath='../Models/WindNet/CrossValidation/'+str(ch_filter)+'/CV_'+str(cross_valid)+'/SDOPred'+str(history)+str(delay)+'/'
    # if os.path.exists(GCsavepath+'GC_train.npy'):
    #     print "GC exists!"
    #     return 
    #------------------------------------------------------>
    #If the data is new, mask has not been done. Mask file must be run.
    trainpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_%d/Train/*.npy'%cross_valid))
    testpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_%d/Test/*.npy'%cross_valid))
    validationpaths = sorted(glob('../Data/CrossValidation'+str(ch_filter)+'/CV_%d/Test/*.npy'%cross_valid))

    #--------------------
    #Mask path
    #trainpaths_Mask = sorted(glob('../Data/Mask/CrossValidation'+str(ch_filter)+'/CV_'+str(cross_valid)+'/Train/*.npy'))
    #testpaths_Mask = sorted(glob('../Data/Mask/CrossValidation'+str(ch_filter)+'/CV_'+str(cross_valid)+'/Test/*.npy'))
    #--------------------

    xtrain,ytrain,ytrain_stddev = BifurcatedDataloader(trainpaths,param,param_stddev,inshape,True,channelNo)
    xtest,ytest,ytest_stddev = BifurcatedDataloader(testpaths,param,param_stddev,inshape,True,channelNo)
    xval,yval,yval_stddev = BifurcatedDataloader(validationpaths,param,param_stddev,inshape,True,channelNo)
    print "Data loading: Done"
    #Data normlization.
    xmean = np.mean(xtrain) #for AIA.
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
    print "Data Normalization: Done"
    #------------------------------------------------------>
    #Explained in Technicalities.md
    xtest,ytest,ytest_stddev = DataWindowParser(xtest,ytest,ytest_stddev,history,delay)
    xval,yval,yval_stddev = DataWindowParser(xval,yval,yval_stddev,history,delay)

    xtrain = np.reshape(xtrain,[-1,20,isize,isize,n_channel])
    xtest = np.reshape(xtest,[-1,isize,isize,n_channel])
    print "Data preprocessing: Done"
    #------------------------------------------------------>
    print "Model definition"
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.reset_default_graph()
    '''
        As explained in the class file, to create a new instance of WindNet, the following are needed:
        1. History.
        2. Delay.
        3. Placeholders for image data, output data, stddev of output data, and history of output data.
        4. Placeholders for embedding,dropout and learning rate.
        5. A tf Session.
        6. No. of hidden units.
        7. Path to the saved model (if needed), and path to GoogLeNet weights (if needed).

    '''
    x = tf.placeholder(tf.float32,[None,isize,isize,n_channel])
    y = tf.placeholder(tf.float32,[None,n_out])
    y_stddev = tf.placeholder(tf.float32,[None,n_out])
    y2 = tf.placeholder(tf.float32,[None,history,n_out])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)
    xprime = tf.placeholder(tf.float32,[None,history,nc])
    sess = tf.Session()
    hidden_units = 400
    #Declare instance of the model.
    WindModel = WindNet(history,delay,lr,x,y,y_stddev,y2,keep_prob,xprime,sess,hidden_units,path_for_model)
    #Build the netowrk.
    WindModel.BuildNetwork()
    print xtrain.shape
    #Normalize the embeddings.
    WindModel.EmbeddingNormalization(xtrain)
    print "Model definition: Done"
    #------------------------------------------------------>
    '''
        Next we do some manipulation. See, we would not prefer to make a forward through GoogLeNet for every data point, for every training epoch.
        Hence, we do all the forward pass (on training data, and the current testing data) once, and store the embeddings, to be passed to the LSTM.

    '''
    #Save all the embeddings.
    FCtrain = []
    for i in xrange(xtrain.shape[0]):
        FCtrain.append(sess.run(WindModel.FClayer,feed_dict = {x:xtrain[i,:,:,:,:]}))
    FCtrain = np.asarray(FCtrain)
    FCtest = np.reshape(sess.run(WindModel.FClayer,feed_dict = {x:xtest}),[-1,history,WindModel.endshp])

    #Data window paser.
    xtrain_condensed,ytrain_condensed_stddev,ytrain_condensed = DataWindowParser(FCtrain,ytrain_stddev,ytrain,history,delay)
    xo2=[]
    for i in xrange(FCtrain.shape[0]):
    	for k in xrange(FCtrain.shape[1]-delay-history+1):
    		xo2.append(xtrain[i,k:k+history,:])
    xtrain_historied = np.reshape(np.asarray(xo2),[-1,history,isize,isize,n_channel])
    xtrain_condensed = np.reshape(xtrain_condensed,[-1,history,nc])
    del FCtrain
    del FCtest 
    del ytrain_stddev
    del ytrain 
    del xtest
    del ytest
    del ytest_stddev
    del xval
    del yval 
    del yval_stddev
    
    print "Training data reshaped!"
    xt =np.reshape(xtrain_historied,[-1,history,224,224,3])
    del xtrain_historied
    GC_total=[]
    WindModel.GradCam()
    Prediction2 = np.reshape(sess.run(WindModel.Regression, feed_dict = {keep_prob: 1.0,xprime:xtrain_condensed,y:ytrain_condensed}),[-1,n_out])
    for it in xrange(xt.shape[0]):
        #Iterate through all points in the dataset.
        tmp=np.reshape(xt[it,:,:,:,:],[-1,224,224,3])
        #Get embeddings
        Emb = WindModel.GetEmbedding(tmp)
        #Reshape embedding appropriately.
        Emb = np.reshape(Emb,[-1,history,WindModel.endshp])
        '''
            Grad cam can be looked at in the raw form, or with some transformation. Negative grad cam values don't make
            much sense, so while we can understand the negative values to contribute to reducing the regressed value,
            I provide space for open interpretation here. 

            I use the positive values to understand the trend. Negative, absolute
            scaled and squared values are provided for completeness.
        '''
        Map = WindModel.GetGradCam(tmp,Emb)
        Map=(np.abs(Map)+Map)/2.0
        GC_total.append(Map)
    GC_total = np.asarray(GC_total)
    Large_ind = np.where(DeNormImage(Prediction2,ymin,ymax)>=500.0)[0]
    Small_ind = np.where(DeNormImage(Prediction2,ymin,ymax)<=350.0)[0]
    Inds = {"Large":Large_ind, "Small":Small_ind}
    GCT={"GC":GC_total,'IND':Inds}
    np.save(GCsavepath+'GC_train.npy',GCT)
def Pathfinder(history,delay,ch_filter,cross_valid):
    a=[]
    #The base path should be entered as such. Update your base path.
    #bp='../Models/WindNet/FP32_'+str(ch_filter)+'/SDOPred'+str(history)+str(delay)+'/'
    bp='../Models/WindNet/CrossValidation/'+str(ch_filter)+'/CV_'+str(cross_valid)+'/SDOPred'+str(history)+str(delay)+'/'
    #Use below for 3channel image

    #bp='../Models/WindNet/HMI_Final'+'/SDOPred'+str(history)+str(delay)+'/'
    file_list=sorted(glob(bp+'Model_save/model_no*.index'))
    for name in file_list:
        a.append(int(name.split(".")[-2].split("-")[-1]))
    ValidCorr=np.loadtxt(bp+'AE_Predict_ValidCorr'+str(history)+str(delay)+'.txt')
    ValidMSE=np.loadtxt(bp+'AE_Predict_ValidError'+str(history)+str(delay)+'.txt')
    ValidChi2=np.loadtxt(bp+'AE_Predict_ValidChi2'+str(history)+str(delay)+'.txt')
    considered_pts=[]
    for i in a:
        considered_pts.append(ValidCorr[i])
    considered_pts=np.asarray(considered_pts)
    it_no = a[np.where(considered_pts==np.max(considered_pts))[0][0]]
    path_for_model = bp+'Model_save/model_no-'+str(it_no)
    return path_for_model

ch_filter = int(sys.argv[1])
for history in np.arange(1,5):
    for delay in np.arange(1,5):
        for cross_valid in np.arange(1,6):
            path_for_model=Pathfinder(history,delay,ch_filter,cross_valid)
            ModelRestore(history,delay,10,path_for_model,ch_filter,cross_valid)

    
