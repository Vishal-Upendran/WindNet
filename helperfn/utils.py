import numpy as np
import cv2
import random
from glob import glob
import sys
import os

'''
This file here contains helper functions for preprocessing and saving the data.
These functions are all used by the different models- benchmark of not. Each function
shall be explained in-situ.
'''

def corr(x,y,dem='valid'):
    '''
        Funtion to calculate the correlation between x and y. Uses np.correlate() to perform the same.
        Inputs: x: a 1-d float array.
                y: a 1-d float array, same length as x.
                dem: optional, uses 'valid' as default. Can give 'valid', 'full' or 'same'. Refer to
                     np.convolve docstring for details.
        Returns: correlation between x and y. Can be a single float or a 1-d array.
    '''
    x = (x-np.mean(x))/(np.std(x)*len(x))
    y = (y-np.mean(y))/(np.std(y))
    return np.correlate(x,y,dem)

def NormalizeImage(x,xmin=0,xmax=1):
    ''''
        Standard normalization. Performs (x-xmin)/xmax. Does not work if xmax = 0.
        Inputs: x: A numpy array. Tensorflow tensors can be used iff xmin, xmax are also tensors.
                xmin, xmax: Must be broadcastable to x.
        Returns: Normalized x .
    '''
    t = (x-xmin)/(xmax)
    return t

def DeNormImage(x,xmin=0,xmax=1):
    '''
        Reverses normalization. Performs x*xmax+xmin. Does not work if xmax = 0.
        Inputs: x: A numpy array. Tensorflow tensors can be used iff xmin, xmax are also tensors.
                xmin,xmax: Must be broadcastable to x.
        Returns: Normalization reversed x.
    '''
    t= x*(xmax)+xmin
    return t

def Imresize(img, size2=224):
    img2 = []

    '''
        Resizes the image to [size2,size2], and makes a 3-channel image out of the single channel image presented.
        Inputs: img: shape [batch,isize,isize],  where isize for our case is 128.
                size2: size to be resized to. Should be an integer.
        Returns: an image of shape [bacth,size2,size2,3]
    '''
    for i in xrange(img.shape[0]):
        tmp = cv2.resize(img[i,:,:].astype(np.float64),(size2,size2))
        tmp = cv2.merge((tmp,tmp,tmp))
        img2.append(tmp)
    return np.asarray(img2)

def GenOutputdat(data,indexlist):
    '''
        Generates the output dataset from the existing complete solar wind data.
        Inputs: data: the solar wind data of shape [batch_size,no_of_parameters_in_dataset]
                indexlist: list of integers corresponding to indices of the parameters required.
        Returns: a numpy array of shape [batch_size,no_of_required_parameters]
    '''
    for v in indexlist:
        try:
            tmp = np.hstack((tmp,data[:,v]))
        except:
            tmp = data[:,v]
    return np.reshape(tmp,[-1,len(indexlist)])

def SavingTestData(tv,ytest,ymin,ymax,err,path,name,iter_no = 0):
    '''
        Function to save the testing data every iteration.
        Inputs: tv: Predictions on testing dataset. Same shape as ytest.
                ytest: Testing dataset, numpy array.
                ymin,ymax: Data statistics for reversing normalization. Must be broadcastable if array.
                err: numpy array, same shape as ytest. Gives stddev for each element in ytest.
                path: global path for saving the data.
                name: Model name used. Ex: 'RBF'  for RBF SVM.
                iter_no: Iteration number of type int.
        Returns: A matrix containing Testing, Prediction.
    '''
    tv = np.hstack((tv,ytest))
    Prediction = DeNormImage(tv,ymin,ymax)
    localpath = path +'Testing/'
    if not os.path.isdir(localpath):
        os.makedirs(localpath)
    Data={'data':Prediction,'stddev':err}
    np.save(localpath+name+'_Predict_Test_Iter_no'+str(iter_no),Data)
    return Prediction

def SavingTrainingData(tv2,yo,ymin,ymax,err,path,name,iter_no = 0):
    '''
        Function to save the testing data every iteration.
        Inputs: tv: Predictions on training dataset. Same shape as ytest.
                yo: Training dataset, numpy array.
                ymin,ymax: Data statistics for reversing normalization. Must be broadcastable if array.
                err: numpy array, same shape as ytest. Gives stddev for each element in yo.
                path: global path for saving the data.
                name: Model name used. Ex: 'RBF'  for RBF SVM.
                iter_no: Iteration number of type int.
        Returns: A matrix containing Testing, Prediction.
    '''
    tv2 = np.hstack((tv2,yo))
    Prediction2 = DeNormImage(tv2,ymin,ymax)
    localpath = path +'Training/'
    if not os.path.isdir(localpath):
        os.makedirs(localpath)
    Data={'data':Prediction2,'stddev':err}
    np.save(localpath+name+'AE_Predict_Train_Iter_no'+str(iter_no),Data)
    return Prediction2

def chi2(x,y,yerr):
    '''
        Calculates Reduced mean square error.
        Inputs: x,y: numpy array of same shape, to be compared.
                yerr: Error associated per point of observation.
        Returns: reduced mean square error. One float number.
    '''
    return np.mean(np.divide(np.square(x-y),yerr))

def BifurcatedDataloader(filepath,param,param_stddev,inshape,x_needed = True,channelNo=4):
    '''
        Function to load the data from a file. Loads the clean data from bifurcated dataset.
        Inputs: filepath- for data to be loaded. Must be a list of file paths of data.
                param: parameter index no for the output data; list of integers
                param_stddev: corresponding parameter index for the pointwise standard deviation; list of integers
                inshape: shape of the input data.
                x_needed: boolean True or False. Is the x part of input needed?
                channelNo: channelNo required from the 8-channel dataset. Integer.
        Returns: xvalue: the input set (numpy array).
                 yvalue:  the output set (numpy array).
                 ystddev: the stddev associated with each output datapoint (numpy array).
    '''
    xin = []
    yout = []
    yout_stddev = []
    for fname in filepath:
        fopen = np.load(fname,allow_pickle=True)
        fopen = fopen.tolist()
        if x_needed is True:
            din = fopen['input']
            din = np.reshape(din,[-1,inshape[0],inshape[1],inshape[-1]])[:,:,:,channelNo]
            din = Imresize(din,224)
            xin.append(np.reshape(din,[-1,din.shape[-1]*din.shape[-2]*din.shape[-3] ]))
        else:
            pass
        dout = fopen['output']
        yout.append(GenOutputdat(dout,[param]))
        yout_stddev.append(GenOutputdat(dout,[param_stddev]))
    if x_needed is True:
        xin = np.asarray(xin)
    else:
        pass
    yout = np.asarray(yout)
    yout_stddev = np.asarray(yout_stddev)

    return xin,yout,yout_stddev
def  DataWindowParser(xo,yo,ystd,history,delay):
    '''
        Our dataset is of the form [global_batches,batch_size,size]. We need to take the inputs corresponding
        to take a window of length *history*, and map it to an observation at *history+delay*. Hence, we need to keep
        moving this window, and derive a new dataset from the existing data for an easy mapping. The same has been
        explained in the README associated with the utils file.
        Inputs: xo: input data- a numpy array of shape [global_batches,batch_size,...].
                yo: output data - a numpy array of shape [global_batches,batch_size,...].
                ystd: stddev associated with output data - a numpy array of shape [global_batches,batch_size,...].
                history, delay: history and delay associated with the model- int.
        Returns: xin,yout,ystd_out: numpy arrays corresponding to input, output and stddev datasets of shape [global_batches,sequential_batch_size,...]
    '''
    xin=[]
    yout=[]
    ystd_out=[]
    for i in xrange(xo.shape[0]):
        for j in xrange(xo.shape[1]-delay-history+1):
            xin.append(xo[i,j:j+history,:])
            yout.append(yo[i,j+history+delay-1,:])
            ystd_out.append(ystd[i,j+history+delay-1,:])
    xin = np.asarray(xin)
    yout = np.asarray(yout)
    ystd_out = np.asarray(ystd_out)
    return xin,yout,ystd_out