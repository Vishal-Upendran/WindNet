import os
from shutil import rmtree,copyfile,copy2
from glob import glob
import sys 

channelNo= int(sys.argv[1])
mask=True  
if mask:
	basepath='../Data/Mask/CrossValidation'+str(channelNo)
else:
	basepath='../Data/CrossValidation'+str(channelNo)
for i in xrange(1,6):
    cvpath=basepath+'/CV_'+str(i)+'/'
    test=cvpath+'Test/'
    train=cvpath+'Train/'
    if not os.path.isdir(cvpath):
        os.makedirs(test)
        os.makedirs(train)
    else:
        rmtree(cvpath)
        os.makedirs(test)
        os.makedirs(train)
    for j in xrange(1,6):
        foldpath=glob(basepath+'/Fold'+str(j)+'/*')
        if j!=i:
            for file in foldpath:
                copy2(file,train)
        else:
            for file in foldpath:
                copy2(file,test)