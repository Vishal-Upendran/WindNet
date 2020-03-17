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
print ch_filter
#----
'''
    Define the constants first.
'''
parser = argparse.ArgumentParser(description = 'Rebuilding WindNet model')
parser.add_argument('history',type = int,help = 'History parameter')
parser.add_argument('delay',type = int, help = 'Delay parameter')
parser.add_argument('n_iter', type = int, help = 'Iteration number for the model to be selected')
parser.add_argument('basePath', default = 'None',help = 'Custom path for the trained model weights')
parser.add_argument('ch_filter',default = 'None',help = 'Filter number associated with the model')
parser.add_argument('cross_valid',type = int, default = 'None',help = 'Cross validation index')
args = parser.parse_args()
'''
    The parameters have all been explained in Technicalities.md
'''
#isize = 224
#n_channel = 3

#isize_before_pretrained = 128 #Was 128
#n_channel_before_pretrained = 1
#channelNo = 4 #Was 4
#n_in = isize*isize*n_channel
#n_in_before_pretrained = isize_before_pretrained*isize_before_pretrained*n_channel_before_pretrained
learning_rate = 0.0001 #Was 0.00001 for others
kp=0.55
#n_out = 1
#param = 11 #Was 11
#param_stddev = param+3 #Was param+3
history = args.history#int(sys.argv[2])
delay = args.delay#int(sys.argv[1])
n_iter2 = args.n_iter#int(sys.argv[3])
ch_filter=args.ch_filter
cross_valid = args.cross_valid
print ch_filter
Direc='/tmp/tenorflow/run9'
nc = 832
#inshape=[isize_before_pretrained,isize_before_pretrained,8]
print "Delay: " + str(delay)
print "History: " + str(history)
print "N_iterations: " + str(n_iter2)

#Path for saving the predictions, if needed.
path = '../ForPlotting/WindNet/SDOPred'+str(history)+str(delay)+'/'
if args.basePath == 'None':
    path_for_model = path+'Model_save/'+'model_no'+str(n_iter2)
else:
    path_for_model = args.basePath

print path_for_model
#------------------------------------------------------>
print "Data loading started"

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
#-------------------
#Data path
#trainpaths = sorted(glob('../Data/Bifurcated_data_'+str(ch_filter)+'/Train/*.npy')) 
#testpaths = sorted(glob('../Data/Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))
#validationpaths = sorted(glob('../Data/Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))

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
#------------------------------------------------------>
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
print "Training data reshaped!"
#------------------------------------------------------>
'''
    Model loading, and dataset preparation is done. Use them in your model now!
'''
print "Model loading, and dataset preparation is done. Use them in your model now!"
