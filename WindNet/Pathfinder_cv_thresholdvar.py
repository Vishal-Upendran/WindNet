import sys
import os
from glob import glob
import numpy as  np
'''
    A script to find the path to model
'''
history = int(sys.argv[1])
delay = int(sys.argv[2])
ch_filter = int(sys.argv[3])
cross_valid=int(sys.argv[4])
tv1=int(sys.argv[5])
tv2=int(sys.argv[6])
a=[]
#The base path should be entered as such. Update your base path.
#bp='../Models/WindNet/FP32_'+str(ch_filter)+'/SDOPred'+str(history)+str(delay)+'/'
#bp='../Models/WindNet/CrossValidation/'+str(ch_filter)+'/CV_'+str(cross_valid)+'/SDOPred'+str(history)+str(delay)+'/'
bp = '../Models'+'_'+str(tv1)+'_'+str(tv2)+'/WindNet/CrossValidation/'+str(ch_filter)+'/CV_'+str(cross_valid)+'/SDOPred'+str(history)+str(delay)+'/'
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
Maxcorr=np.max(considered_pts)
VChi2=ValidChi2[it_no]
VMSE=ValidMSE[it_no]

'''
	Or simply replace the path_for_model with the path to appropriate model.
'''
