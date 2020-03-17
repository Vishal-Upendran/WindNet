import numpy as np
import cv2
from glob import glob
import sys
import os
#-------------
from skimage.exposure import equalize_hist as eqHist
from skimage.filters import gaussian as Gauss
import matplotlib.gridspec as gridspec
from skimage.filters import threshold_otsu as Otsu
from sklearn.mixture import GaussianMixture as GMM
#-------------
from helperfn.utils import *
#-------------
#cv = int(sys.argv[1])

#Module to generate the Active regions and  Coronal holes mask.
#We use the same masking algorithm for both the 193 AA and 211 AA channels.

def GetCorhole(sample,mask):
    '''
        This function segments out the coronal holes from our images. It uses Otsu thresholding with morphological
        operations to extract them out.
        Inputs:
            sample: img of shape [isize,isize], minvalue = 0 and maxvalue = 255
            mask: binary mask of shape [isize,isize] to segment out only the solar disc.
    '''
    #sample = cv2.GaussianBlur(sample,(5,5),10)
    #Initial smoothing
    delta = 0.03
    sample = cv2.bilateralFilter(sample.astype(np.float32),5,75,75)
    #Thresholding
    val = Otsu(sample)
    th = (sample*mask)<=(val+delta)
    th = th*mask
    #Approximate clean up
    #kernel = np.ones((7,7),np.uint8)
    #th = cv2.dilate(th.astype(np.uint8),kernel,iterations = 1)
    #kernel = np.ones((6,6),np.uint8)
    #th = cv2.erode(th.astype(np.uint8),kernel,iterations = 3)
    val = Otsu(th*sample)
    th2 = (th*sample)<=(val+delta)
    return th*th2

def GetHotAreas(img,mask):
    '''
        This function segments out the active regions from our images. It uses a Gaussian Mixture Model (GMM)
        to segement out the ARs. GMM can be understood to be a generalization of Otsu thresholding.
        Inputs:
            img: img of shape [isize,isize], minvalue = 0 and maxvalue = 255
            mask: binary mask of shape [isize,isize] to segment out only the solar disc.
    '''
    inp=img*mask
    #Initial smoothing
    inp = cv2.bilateralFilter(inp.astype(np.float32),9,75,75)
    #Define the mixture model, and take the component with highest mean value.
    gmodel = GMM(n_components=5)
    gmodel.fit(np.reshape(inp,[-1,1]))
    th_gmm = gmodel.predict(np.reshape(inp,[-1,1]))
    th_gmm = th_gmm == np.where(np.asarray(gmodel.means_)==np.max(gmodel.means_))[0]
    return np.reshape(th_gmm,[224,224])

def AutoNormalize(img):
    #Automatically normalize between 0 and 1
    img =(img-np.min(img))/(np.max(img)-np.min(img))
    return img
#Our mask is a disc with 1 on the solar disc and 0 outside
mask = np.zeros([224,224])
for i in xrange(224):
    for j in xrange(224):
        if np.square(i-112)+np.square(j-112)<=90*90:
            mask[i,j]=1
#-----------------------------------------------------------
isize = 224
n_channel = 3

isize_before_pretrained = 256 #Was 128
n_channel_before_pretrained = 1

#------------
#ch_filter is either 193 or 211. It must be defined before importing this module.
ch_filter = int(sys.argv[1])
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
for cv in np.arange(1,6):
    #Data path
    #trainpaths = sorted(glob('Data/Bifurcated_data_'+str(ch_filter)+'/Train/*.npy'))
    #testpaths = sorted(glob('Data/Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))
    trainpaths = sorted(glob('Data/CrossValidation'+str(ch_filter)+'/Fold'+str(cv)+'/*.npy'))
    #testpaths = sorted(glob('Data/CrossValidation'+str(ch_filter)+'/Fold'+str(cv)+'/Test/*.npy'))
    #print testpaths
    ep=1e-5
    #All file paths are here.
    for file in trainpaths:
        din = np.load(file,allow_pickle=True).tolist()['input']
        din = Imresize(np.reshape(din,[-1,isize_before_pretrained,isize_before_pretrained]),224)[:,:,:,0]
        CH_mask=[]
        AR_mask=[]
        for i in xrange(din.shape[0]):
            tmp=AutoNormalize(din[i,:,:])
            ch=GetCorhole(tmp,mask)
            ar=GetHotAreas(tmp,mask)
            CH_mask.append(ch*1.0/(np.sum(ch)+ep))
            AR_mask.append(ar*1.0/(np.sum(ar)+ep))
        Mask = {'hole':np.asarray(CH_mask),'hot':np.asarray(AR_mask)}
        file_name = file.split('/')[-1]
        #np.save('Data/Mask/Bifurcated_data_'+str(ch_filter)+'/Train/'+file_name,Mask)
        mpath = 'Data/Mask/CrossValidation'+str(ch_filter)+'/Fold'+str(cv)+'/'
        if not os.path.isdir(mpath):
            os.makedirs(mpath)
        np.save(mpath+file_name,Mask)

    #for file in testpaths:
    #    din = np.load(file,allow_pickle=True).tolist()['input']
    #    din = Imresize(np.reshape(din,[-1,isize_before_pretrained,isize_before_pretrained]),224)[:,:,:,0]
    #    CH_mask=[]
    #    AR_mask=[]
    #    for i in xrange(din.shape[0]):
    #        tmp=AutoNormalize(din[i,:,:])
    #        ch=GetCorhole(tmp,mask)
    #        ar=GetHotAreas(tmp,mask)
    #        CH_mask.append(ch*1.0/np.sum(ch))
    #        AR_mask.append(ar*1.0/np.sum(ar))
    #    Mask = {'hole':np.asarray(CH_mask),'hot':np.asarray(AR_mask)}
    #    file_name = file.split('/')[-1]
    #    #np.save('Data/Mask/Bifurcated_data_'+str(ch_filter)+'/Test/'+file_name,Mask)
    #    mpath = 'Data/Mask/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/'
    #    if not os.path.isdir(mpath):
    #        os.makedirs(mpath)
    #    np.save('Data/Mask/CrossValidation'+str(ch_filter)+'/CV_'+str(cv)+'/Test/'+file_name,Mask)
