#import ipywidgets as widgets
#from IPython.display import display
from glob import glob 
import argparse
import os
import sys

from helperfn.utils import *

parser = argparse.ArgumentParser(description = 'Rebuilding WindNet model')
parser.add_argument('history',type = int,help = 'History parameter')
parser.add_argument('delay',type = int, help = 'Delay parameter')
parser.add_argument('basePath', default = 'None',help = 'Custom path for the trained model weights')
parser.add_argument('cv',type=int)
parser.add_argument('bp',type=str)
args = parser.parse_args()

history = args.history#int(sys.argv[2])
delay = args.delay#int(sys.argv[1])
path_for_model = args.basePath
bp=args.bp
cv=args.cv

#bp = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0] #Obtain the path of file above; i.e ../BenchmarkModels
if path_for_model=='NaiveMean':
    path =  bp+'Models/NaiveMean/193/CV_'+str(cv)+'/Persist'+str(history)+str(delay)+'/'
    datapath = path+'/Testing/NaiveMean_Predict_Test_Iter_no0.npy'
    Data = np.load(datapath,allow_pickle=True).tolist()
    Error = Data['stddev']
    Prediction = Data['data']
elif path_for_model=='Persist':
    path = bp+'Models/Persistence/211/CV_'+str(cv)+'/Persist'+str(history)+str(delay)+'/'
    datapath = path+'/Testing/Persist_Predict_Test_Iter_no0.npy'
    Data = np.load(datapath,allow_pickle=True).tolist()
    Error = Data['stddev']
    Prediction = Data['data']
elif path_for_model.split('_')[0]=='SWSVM':
    path = bp+'Models/SVM_SW/193/CV_'+str(cv)+'/SVM_SW'+str(history)+str(delay)+'/'
    datapath = path + '/Testing/'+path_for_model.split('_')[-1]+'_Predict_Test_Iter_no0.npy'
    Data = np.load(datapath,allow_pickle=True).tolist()
    Error = Data['stddev']
    Prediction = Data['data']
elif path_for_model.split('_')[0]=='XGBoost':
    path = bp+'Models/XGBoost_SW/193/CV_'+str(cv)+'/XGBoost_SW'+str(history)+str(delay)+'/'
    datapth = path +'/Testing/'+path_for_model.split('_')[0]+'_Predict_Test_Iter_no0.npy'
    Data = np.load(datapth,allow_pickle=True).tolist()
    Error = Data['stddev']
    Prediction = Data['data']
else:
    raise ValueError('Enter a valid model type from the list given in the previous cell')


