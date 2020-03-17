import numpy as np
from glob import glob
import sys
import os
import shutil
import cv2
#Channel no of AIA to process.
img_size=256
tv1=500
tv2=20000
def ProcessHMI(x):
	'''
	The pre-preprocessing routine for the new images from LMSAL. These images have been sourced from
	/bigdata/FDL/AIA/211/*, on learning.lmsal_gpu.internals.
	The images need to be first log-scaled, and low-clipped at log(10). High clipping at log(10000).
	Finally, rescaling from 0 to 256.
	'''
	tmp = x
	minv=np.min(tmp)
	maxv=np.max(tmp)
	tmp = (tmp-minv)/(maxv-minv)
	zeroval = -minv/(maxv-minv)
	tmp = tmp*255.0
	tmp = tmp - zeroval*255.0
	return tmp
def ProcessImg(x,channelNo):
	'''
	The pre-preprocessing routine for the new images from LMSAL. These images have been sourced from
	/bigdata/FDL/AIA/211/*, on learning.lmsal_gpu.internals.
	The images need to be first log-scaled, and low-clipped at log(10). High clipping at log(10000).
	Finally, rescaling from 0 to 256.
	'''
	#First check if the image shape is correct.
	shp=x.shape[0]
	if shp!=256:
		x=cv2.resize(x.astype(np.float64),(img_size,img_size))
	tmp = np.log10(x+1e-5)
	minv = np.min(x)
	if channelNo==211:
		threshval=100.0
		satval=10000.0
	if channelNo==193:
		threshval=tv1 #Was 500
		satval=tv2 #Was 20000
	if channelNo==171:
		threshval=800.0
		satval=18000.0
	tmp[np.where(tmp<np.log10(threshval+minv))] = np.log10(threshval+minv) #100 removes a lot of intensity and shows coronal holes well. But will not work for 193
	tmp[np.where(tmp>np.log10(satval+minv))] = np.log10(satval+minv)
	min2 = np.min(tmp)
	max2 = np.max(tmp)
	tmp = (tmp-min2)*256.0/(max2-min2)
	return tmp
def ProcessImg2(x,channelNo):
	'''
	The pre-preprocessing routine for the new images from LMSAL. These images have been sourced from
	/bigdata/FDL/AIA/211/*, on learning.lmsal_gpu.internals.
	The images need to be first log-scaled, and low-clipped at log(10). High clipping at log(10000).
	Finally, rescaling from 0 to 256.
	'''

	'''
		The 2015,2016,2017,2018 data have a different dynamic range than 2011-2014, since the earlier were downloaded from Stanford online repo and the latter from
		LMSAL gpu. Hence, we scale the threshold and saturation by the ratio of range of Dec 31 2014 and Jan 01 2015 data - i.e, we divide by 3.639
	'''
	#First check if the image shape is correct.
	shp=x.shape[0]
	if shp!=256:
		x=cv2.resize(x.astype(np.float64),(img_size,img_size))
	tmp = np.log10(x+1e-5)
	minv = np.min(x)
	if channelNo==211:
		threshval=100.0/4
		satval=10000.0/4
	if channelNo==193:
		threshval=tv1/4.0 #Was 500/4.0
		satval=tv2/4.0 #Was 20000/4.0 #3.639
	if channelNo==171:
		threshval=800.0
		satval=18000.0
	tmp[np.where(tmp<np.log10(threshval+minv))] = np.log10(threshval+minv) #100 removes a lot of intensity and shows coronal holes well. But will not work for 193
	tmp[np.where(tmp>np.log10(satval+minv))] = np.log10(satval+minv)
	min2 = np.min(tmp)
	max2 = np.max(tmp)
	tmp = (tmp-min2)*256.0/(max2-min2)
	return tmp
def ProcessImg_common(x,channelNo):
	'''
	The pre-preprocessing routine for the new images from LMSAL. These images have been sourced from
	/bigdata/FDL/AIA/211/*, on learning.lmsal_gpu.internals.
	The images need to be first log-scaled, and low-clipped at log(10). High clipping at log(10000).
	Finally, rescaling from 0 to 256.
	'''

	'''
		The 2015,2016,2017,2018 data have a different dynamic range than 2011-2014, since the earlier were downloaded from Stanford online repo and the latter from
		LMSAL gpu. Hence, we scale the threshold and saturation by the ratio of range of Dec 31 2014 and Jan 01 2015 data - i.e, we divide by 3.639 ~ 4.0
		The data gaps were plugged by using the data from online repo, thus the existing data correction has been incorporated.
		This occurs since online data is (512,512), and the LMSAL system is (256,256), averaging over 4 pixels.
	'''
	#First check if the image shape is correct.
	shp=x.shape[0]
	if shp!=256:
		x=cv2.resize(x.astype(np.float64),(img_size,img_size))
	else:
		x/=4.0
	tmp = np.log10(x+1e-5)
	minv = np.min(x)
	if channelNo==211:
		threshval=100.0/4
		satval=10000.0/4
	if channelNo==193:
		threshval=500/4.0 #Was 500/4.0
		satval=20000/4.0 #Was 20000/4.0 #3.639
	if channelNo==171:
		threshval=800.0
		satval=18000.0
	tmp[np.where(tmp<np.log10(threshval+minv))] = np.log10(threshval+minv) #100 removes a lot of intensity and shows coronal holes well. But will not work for 193
	tmp[np.where(tmp>np.log10(satval+minv))] = np.log10(satval+minv)
	min2 = np.min(tmp)
	max2 = np.max(tmp)
	tmp = (tmp-min2)*256.0/(max2-min2)
	return tmp

img_type = 'AIA'
channelNo = int(sys.argv[1])
years = ['2011','2012','2013','2014','2015','2016','2017','2018']
mnth = ['%.2d' % i for i in range(1,13)]
dy = ['%.2d' % i for i in range(1,32)]
bp = '../'+img_type+'/'+str(channelNo)+'/'
save_bp='../Processed_data/'+str(channelNo)+'/'
if not os.path.isdir(save_bp):
	os.makedirs(save_bp)
else:
	shutil.rmtree(save_bp)
	os.makedirs(save_bp)
for yr in years:
	print yr+" started: "
	for mn in mnth:
		for d in dy:
			try:
				path = sorted(glob(bp+yr+'/'+mn+'/'+d+'/*'))[0]
				# if yr in ['2011','2012','2013','2014']:
				image = ProcessImg_common(np.load(path)['x'],channelNo)
				# else:
				# 	image = ProcessImg2(np.load(path)['x'],channelNo)
				#print np.load(path)['x'].shape 
				np.save(save_bp+'_'+yr+mn+d+'.npy',image)
			except:
				pass
	print yr+" done."
