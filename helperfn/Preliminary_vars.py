'''
	Some variables, like the path for data, initial input size and channel no are used by multiple scripts. 
	We define them all here to maintain consistency across all scripts.
'''
from glob import glob
import os 

isize = 224
n_channel = 3

isize_before_pretrained = 256 
n_channel_before_pretrained = 1

#------------
#ch_filter is either 193 or 211. It must be defined before importing this module.
ch_filter = 193
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
#-------------------
#Data path
trainpaths = sorted(glob('../Data/Bifurcated_data_'+str(ch_filter)+'/Train/*.npy'))
testpaths = sorted(glob('../Data/Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))
validationpaths = sorted(glob('../Data/Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))

#If it is new data, mask has not been done. Run masking file.
#trainpaths = sorted(glob('../Data/Dattaraj_Bifurcated_data_'+str(ch_filter)+'/Train/*.npy')) 
#testpaths = sorted(glob('../Data/Dattaraj_Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))
#validationpaths = sorted(glob('../Data/Dattaraj_Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))

#--------------------
#Mask path
trainpaths_Mask = sorted(glob('../Data/Mask/Bifurcated_data_'+str(ch_filter)+'/Train/*.npy'))
testpaths_Mask = sorted(glob('../Data/Mask/Bifurcated_data_'+str(ch_filter)+'/Test/*.npy'))

