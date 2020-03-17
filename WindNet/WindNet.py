'''
    This file defines the class object of our network. It can simply be imported and plugged and run later on.
'''
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import sys
import os
from tensorflow.python.framework import ops
from skimage.exposure import equalize_hist as eqHist
from skimage.filters import gaussian as Gauss

from KaffeModelConverted.googlenet import GoogleNet
sys.path.append('../')
# Custom helpers functions.
from helperfn.utils import *
from helperfn.tf_utils import *
#----

'''
Registering modified gradient for epsilon-LRP. Implementation based on:
https://github.com/marcoancona/DeepExplain
'''
@ops.RegisterGradient("ELRP2")
def modified_gradient(op,grad):
    outp = op.outputs[0]
    inp = op.inputs[0]
    return grad*outp/(inp+1e-4)

'''
Map the appropriate function key to the modified gradient. Tensorflow will use this function instead of the ususal gradient
during backpropagation, at every layer
'''
ActivationList = ['Identity']
Dict_grad_override = dict((v,'ELRP2') for v in ActivationList)

'''
Define helpers for modifying backprop.
'''
def symbolic_attribution(outp,inp,sess):
    g=tf.get_default_graph()
    with g.gradient_override_map(Dict_grad_override):
        with sess.as_default():
            grads = tf.stack(tf.gradients(tf.identity(outp),inp))
            changed_grad = grads*inp
            return changed_grad
def Gen_gradient(outp,inp,sess):
    g=tf.get_default_graph()
    with g.gradient_override_map(Dict_grad_override):
        with sess.as_default():
            grads = tf.stack(tf.gradients(tf.identity(outp),inp))
            changed_grad = grads
            return changed_grad

class WindNet:
    '''
        This is the flow if the class defined here:
        1. Initialize all the parameters for the model, and placeholders required for the model.
        2. Define the Embedding generator network graph.
        3. Define the Timeseries regressor graph.
        4. If using pretrained weights, load them.
        5. Normalize the embeddings.
        6. Get the embeddings out.
        7. Any layer needed can be just called out using object_name.layer_name
        8. Add a visualizer class next.
    '''
    def __init__(self,history,delay,learning_rate,x,y,y_stddev,y2,keep_prob,xprime,sess1,hidden_units,path_to_all_weights = None,path_to_google_weights = 'KaffeModelConverted/googlenet.npy'):
        '''
            Initializer for the WindNet class. The initlizer must be provided with:
            1. history: [dtype = int] The history of data to be used. This history is the set of images which will be used
                        as input for regression.
            2. delay: [dtype = int] The delay between the latest SDO image and the predicted data point.
            3. learning_rate: [dtype = float32] The learning rate for training the network. Reccommended value: 0.0001
            4. x: [dtype = tf.placeholder, shape: [batch_size,isize,isize,n_channel]] Placeholder for giving in the input images
                  for embedding.
            5. y: [dtype = tf.placeholder, shape:[batch_size,n_out]] Placeholder for giving the output observation values.
            6. y_stddev: [dtype = tf.placeholder, shape:[batch_size,n_out]] Placeholder for giving the stddev associated with
                         the output observation values.
            7. y2: [dtype = tf.placeholder, shape:[batch_size,history,n_out]] Placeholder for matching sequence of outputs
                   mapped while training using the LSTM.
            8. keep_prob: [dtype = tf.placeholder, shape:[tf.float32]] Placeholder for giving dropout values.
            9. xprime: [dtype = tf.placeholder, shape:[batch_size,history,endshp]]Placeholder to feed in th embeddings.
            10. sess1: [dtype = tf.Session] Tf session to run the code.
            11. hidden_units: [dtype = int]  No. of hidden units in LSTM.
            12. path_to_all_weights: [dtype = string] Path to total saved model.
            13. path_to_google_weights: [dtype = string] Path to GoogleNet saved model.
        '''
        self.history = history
        self.delay = delay
        self.learning_rate = learning_rate
        self.x = x
        self.y = y
        self.y_stddev = y_stddev
        self.y2 = y2
        self.keep_prob = keep_prob
        self.xprime = xprime
        self.sess = sess1
        self.path_to_google_weights = path_to_google_weights
        self.hidden_units = hidden_units
        self.weights= {}
        self.biases = {}
        shape_list = self.x.get_shape().as_list()
        self.n_in = shape_list[-1]*shape_list[-2]*shape_list[-3]
        self.isize = shape_list[-2]
        self.n_channel = shape_list[-1]
        self.n_out = self.y.get_shape().as_list()[-1]
        self.path_to_all_weights = path_to_all_weights

    def BuildNetwork(self):
        '''
            This function builds the network graph. It is to be called right after initializing the
            parameters required for the model. The various steps to model definition shall be explained next:
            1. The pretrained GoogLeNet is loaded, and the relevant layers extracted.
            2. Each layer is preprocessed as FC<---Sum_over_pixels_per_channel(Square_each_pixel(Layer)) .
            3. The LSTM cell is defined with the given no. of hidden units.
            4. For the sake of faster training, the graph has been split as GoogLeNet and LSTM regressor.
               The data has one forward pass done on the GoogLeNet, and the embeddings are stored in RAM.
               These embeddings are then passed on to the LSTM to be trained. While the splitting of graph causes
               some complications, we accept them in lieu of speeding up the forward pass for each training.
            5. The final output term is self. Regression, which must be run for getting the requried predictions.
        '''
        self.Embedder = GoogleNet({'data':self.x},trainable = False)
        self.Embedder.load(self.path_to_google_weights,self.sess)
        self.GNet_layers = self.Embedder.layers
        #We have loaded the GoogLeNet pretrained model with weights here.
        self.FC1= tf.reduce_sum(tf.square(self.GNet_layers['conv2_3x3_reduce']),axis = [1,2])
        self.FC2 = tf.reduce_sum(tf.square(self.GNet_layers['inception_4a_output']),axis = [1,2])
        self.FC3 = tf.reduce_sum(tf.square(self.GNet_layers['inception_3a_output']),axis = [1,2])
        #Getting out the appropriate sizes
        self.shape1 = self.FC1.get_shape().as_list()
        self.shape2 = self.FC2.get_shape().as_list()
        self.shape3 = self.FC3.get_shape().as_list()
        self.endshp = self.shape1[-1]+self.shape2[-1]+self.shape3[-1]
        #Define the LSTM Cell to feed in the embeddings.
        self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_units,use_peepholes=False),output_keep_prob=self.keep_prob)
        self.output,self.state = tf.nn.dynamic_rnn(self.cell, self.xprime,dtype = tf.float32)
        self.output = tf.stack(self.output)
        #state = tf.reshape(tf.transpose(tf.stack(self.state),[1,0,2]),[-1,self.hidden_units*2])
        self.outshape = self.output.get_shape().as_list()
        self.Reg_output_for_training=[]
        for i in xrange(self.history):
            self.weights['LinReg'+str(i)],self.biases['LinReg'+str(i)] = GetFCVariable([self.outshape[-1],self.n_out])
            self.Reg_output_for_training.append(tf.nn.elu(tf.matmul(self.output[:,i,:],self.weights['LinReg'+str(i)])+self.biases['LinReg'+str(i)]))
        self.Reg1 = self.Reg_output_for_training[-1]
        self.Regression = self.Reg1 #tf.nn.elu(tf.matmul(Reg1,weights['LinReg_'])+biases['LinReg_'])
        self.Reg_output_for_training = tf.transpose(tf.stack(self.Reg_output_for_training),[1,0,2])
        #Next, define the metrics, optimizers, etc.
        self.MakeMetricOptimizer()

    def MakeMetricOptimizer(self):
        '''
            This function does the variable initialization and cost definitions. We use a small sleight of hand in our
            cost function:
            For each input given to the LSTM, there is a corresponding output. The successive predictions depend on the
            previous input, and the final output is what we present as our output. However, we try to make our LSTM fit
            to a window of input to a window of output including the presented output. Along with this term, we include
            a term which emphasizes on the importance of the final output - the chi2_reduced metric, which is essentially
            the squared error weighted by the variance associated with the corresponding observation.
            To prevent overfitting, we also include an L2 regularization term and a dropout. Finally, if the path to
            a saved model is given, we use the same for initializing our weights. Else, we go ahead with random initialization.

        '''
        regularization = tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()])
        self.cost= tf.reduce_mean(tf.square(self.Regression-self.y))
        self.cost1= tf.reduce_mean(tf.square(self.Reg_output_for_training-self.y2))
        self.chi2_reduced = tf.reduce_mean(tf.divide(tf.square(self.Regression-self.y),tf.square(self.y_stddev)))
        self.cost_pred = self.cost+0.1*self.cost1+self.chi2_reduced+ 0.000001*regularization
        self.optimizer_pred = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_pred)
        if self.path_to_all_weights is None:
            self.sess.run(tf.initialize_all_variables())
        else:
            self.saver=tf.train.Saver()
            self.saver.restore(self.sess,self.path_to_all_weights)
    def saver_init(self):
        self.saver=tf.train.Saver(max_to_keep = 1000)
    def save_model(self,path):
        _ = self.saver.save(self.sess, path)


    def EmbeddingNormalization(self,xtrain):
        '''
            :IMPORTANT:
            Before using the network inference or training, the embeddings must be normalized. Since we
            take Sum(Square(Pixels)), there could be discrepancies in the due to no. of pixels, and the
            general range of pixel values in the layers. Hence, a forward pass must be done using training data,
            and the resulting raw embeddings normalized. The final embedding to be fed to our LSTM would be
            the normalized embeddings, concatenated along the filter axis.

        '''
        FCtrain = []
        for i in xrange(xtrain.shape[0]):
            FCtrain.append(self.sess.run(self.FC1,feed_dict = {self.x:xtrain[i,:,:,:,:]}))

        self.FCmin1 = np.min(FCtrain,axis = (0,1))
        self.FCmax1 = np.max(FCtrain,axis = (0,1))-self.FCmin1+1e-8
        self.FC11 = NormalizeImage(self.FC1,self.FCmin1,self.FCmax1)

        FCtrain = []
        for i in xrange(xtrain.shape[0]):
            FCtrain.append(self.sess.run(self.FC2,feed_dict = {self.x:xtrain[i,:,:,:,:]}))

        self.FCmin2 = np.min(FCtrain,axis = (0,1))
        self.FCmax2 = np.max(FCtrain,axis = (0,1))-self.FCmin2+1e-8
        self.FC22 = NormalizeImage(self.FC2,self.FCmin2,self.FCmax2)

        FCtrain = []
        for i in xrange(xtrain.shape[0]):
            FCtrain.append(self.sess.run(self.FC3,feed_dict = {self.x:xtrain[i,:,:,:,:]}))

        self.FCmin3 = np.min(FCtrain,axis = (0,1))
        self.FCmax3 = np.max(FCtrain,axis = (0,1))-self.FCmin3+1e-8
        self.FC33 = NormalizeImage(self.FC3,self.FCmin3,self.FCmax3)

        self.FClayer = tf.concat([self.FC11,self.FC22,self.FC33],axis = 1)


    def GetEmbedding(self,indat):
        '''
            Function to return the Embeddings. Lazy people could directly use as Model.FClayer.
        '''
        return self.sess.run(self.FClayer,feed_dict = {self.x: indat,self.keep_prob:1.0})

    def GradCam(self):
        '''
            Given a set of feature maps, and the corresponding size, we will caculate the GRADCAM Map.
            Look at the Techincalities.md file for detailed explanation.
        '''
        Reg_B_grad = tf.reshape(Saliency(self.Regression,self.xprime,self.sess),[-1,1,1,self.endshp])#Output is [batch,shape1]
        print Reg_B_grad.get_shape()
        fmap = self.GNet_layers['conv2_3x3_reduce']
        Partial_B_A = 2.0*fmap
        print Partial_B_A.get_shape()
        Partial_Reg_A = tf.reduce_mean(tf.multiply(Reg_B_grad[:,:,:,:self.shape1[-1]],Partial_B_A),axis=(1,2))
        print Partial_Reg_A.get_shape() #[batch,shape1]
        Partial_Reg_A = tf.reshape(Partial_Reg_A,[-1,1,1,self.shape1[-1]])
        self.GC1 = tf.reduce_mean(tf.multiply(Partial_Reg_A,fmap),axis=3)
        #--------------
        fmap = self.GNet_layers['inception_4a_output']
        Partial_B_A = 2.0*fmap
        print Partial_B_A.get_shape()
        Partial_Reg_A = tf.reduce_mean(tf.multiply(Reg_B_grad[:,:,:,self.shape1[-1]:self.shape1[-1]+self.shape2[-1]],Partial_B_A),axis=(1,2))
        print Partial_Reg_A.get_shape() #[batch,shape1]
        Partial_Reg_A = tf.reshape(Partial_Reg_A,[-1,1,1,self.shape2[-1]])
        self.GC2 = tf.reduce_mean(tf.multiply(Partial_Reg_A,fmap),axis=3)
        #--------------
        fmap = self.GNet_layers['inception_3a_output']
        Partial_B_A = 2.0*fmap
        print Partial_B_A.get_shape()
        Partial_Reg_A = tf.reduce_mean(tf.multiply(Reg_B_grad[:,:,:,self.shape1[-1]+self.shape2[-1]:],Partial_B_A),axis=(1,2))
        print Partial_Reg_A.get_shape() #[batch,shape1]
        Partial_Reg_A = tf.reshape(Partial_Reg_A,[-1,1,1,self.shape3[-1]])
        self.GC3 = tf.reduce_mean(tf.multiply(Partial_Reg_A,fmap),axis=3)
        #--------------

    def GetGradCam(self,indat,embedd):
        '''
            Function to generate the GradCam map. Eseentially runs the Cams in a session,
            and returns the final map per day of input image. While we have the maps from
            each layer, these have different sizes, and must be scaled up to (isize,isize)
            for addition, and for the final GradCam map. This is also done here.
        '''
        Map1 = self.sess.run(self.GC1,feed_dict={self.x:indat,self.xprime:embedd,self.keep_prob:1.0})
        Map2 = self.sess.run(self.GC2,feed_dict={self.x:indat,self.xprime:embedd,self.keep_prob:1.0})
        Map3 = self.sess.run(self.GC3,feed_dict={self.x:indat,self.xprime:embedd,self.keep_prob:1.0})
        Map1 = np.reshape(Map1,[-1,self.history,Map1.shape[-1],Map1.shape[-1]])
        Map2 = np.reshape(Map2,[-1,self.history,Map2.shape[-1],Map2.shape[-1]])
        Map3 = np.reshape(Map3,[-1,self.history,Map3.shape[-1],Map3.shape[-1]])

        GradCamMap = []
        for i in xrange(Map1.shape[0]):
            tmp_map = []
            for j in xrange(self.history):
                m1 = cv2.resize(Map1[i,j,:,:],(224,224),interpolation=cv2.INTER_CUBIC)
                m2 = cv2.resize(Map2[i,j,:,:],(224,224),interpolation=cv2.INTER_CUBIC)
                m3 = cv2.resize(Map3[i,j,:,:],(224,224),interpolation=cv2.INTER_CUBIC)
                m1 = m1+m2+m3
                tmp_map.append(m1)
            tmp_map=np.asarray(tmp_map)
            GradCamMap.append(tmp_map)
        GradCamMap = np.asarray(GradCamMap)
        return GradCamMap
    def Epsilon_LRP(self):
        '''
        Define the eLRP tensors here.
        '''
        self.LRPMap = Gen_gradient(self.Regression,self.xprime,self.sess)
        FC_unst = tf.unstack(self.FClayer,axis = -1)
        InpLRPMap = tf.stack([symbolic_attribution(v,self.x,self.sess) for v in FC_unst])
        self.InpLRPMap2 = tf.transpose(InpLRPMap,[2,1,0,3,4,5])
    def GetEpsilon_LRP(self,indat,embedd):
        '''
        Return the eLRP maps for given inputs.
        '''
        self.Sample_LRP_inp = self.sess.run(self.InpLRPMap2,feed_dict = {self.keep_prob: 1.0,self.x:indat})
        self.Sample_LRP_emb = self.sess.run(self.LRPMap,feed_dict = {self.keep_prob: 1.0,self.xprime:embedd})
        self.elrp = np.mean(self.Sample_LRP_inp[:,0,:,:,:,0]*np.reshape(self.Sample_LRP_emb[0,0,:,:],[-1,self.endshp,1,1]),axis=1)
        return self.elrp
    '''
    Occlusion based visualization technique.
    '''
    def make_mask(self,img,centerx,centery,ksize):
        '''
            Generate the mask at relevant position with relevant size.
        '''
        imshape=img.shape
        mask=np.ones(imshape)
        #Making a square mask only.
        try:
            mask[centerx-ksize:centerx+ksize,centery-ksize:centery+ksize]=0
        except:
            print "Unable to generate mask. Make sure maskcenter+ksize remains inside the image"
        return mask

    def mask_image(self,img,centerx,centery,ksize,mtype='dark'):

        img = img[:,:,0] #Ignoring other channels since we just duplicate the first channel in our problem.
        mask=self.make_mask(img,centerx,centery,ksize)
        tmp=img
        minv=np.min(tmp)
        #Get all the parts except the patch in consideration
        if mtype=='avg':

            tmp2 = np.multiply(mask,tmp)
            mask = 1.0-mask
            tmp1 = (np.sum(np.multiply(mask,tmp))/(ksize*ksize*4))*(mask)
            tmp = tmp1+tmp2
            tmp = np.reshape(Imresize(np.reshape(tmp,[-1,self.isize,self.isize])),[self.isize,self.isize,self.n_channel])
        elif mtype=='dark':
            tmp=np.multiply(mask,tmp)
            tmp = np.reshape(Imresize(np.reshape(tmp,[-1,self.isize,self.isize])),[self.isize,self.isize,self.n_channel])
        elif mtype=='light':
            tmp2 = np.multiply(mask,tmp)
            mask = 1.0-mask
            tmp1 = np.max(tmp)*(mask)
            tmp = tmp1+tmp2
            tmp = np.reshape(Imresize(np.reshape(tmp,[-1,self.isize,self.isize])),[self.isize,self.isize,self.n_channel])
        else:
            tmp=tmp
        return tmp
    def perform_occlusion(self,dataset,centerx,centery,ksize,daylist,mtype):
        dataset=np.reshape(dataset,[-1,self.history,self.isize,self.isize,self.n_channel])
        img=np.reshape(dataset,[self.isize,self.isize,self.n_channel])
        tmp=[]
        for i in xrange(history):
            if i in daylist:
                tmp.append(self.mask_image(img[i,:,:,:],centerx,centery,ksize,mtype))
            else:
                tmp.append(img[i,:,:,:])
        tmp=np.asarray(tmp)
        return tmp
    def UniMask(self,dataset,centerx,centery,ksize,daylist,mtype):
        masked_img = self.perform_occlusion(dataset,centerx,centery,ksize,daylist,mtype)
        Emb = np.reshape(self.sess.run(self.FClayer,feed_dict={self.x:masked_img,self.keep_prob:1.0}),[-1,self.history,self.endshp])
        swvalue=self.sess.run(self.Regression,feed_dict={self.xprime:Emb, self.keep_prob:1.0})
        return swvalue,masked_img[:,:,:,0]
    def SweepMask(self,dataset,ksize,stride,daylist,mtype):
        dataset=np.reshape(dataset,[-1,self.history,self.isize,self.isize,self.n_channel])
        tmp=dataset
        imshape=dataset.shape
        for xind in np.arange(ksize,imshape[-3],stride):
            for yind in np.arange(ksize,imshape[-2],stride):
                swv,im=self.UniMask(tmp,xind,yind,ksize,daylist,mtype)
                try:
                    swvalue.append([xind,yind,swv])
                    imageswept.append(im)
                except:
                    swvalue = [[xind,yind,swv]]
                    imageswept = [im]
        return np.asarray(swvalue),np.asarray(imageswept)
    def GetOcclusionMap(self,dataset,ksize,stride,daylist,outdat,mtype_list = ['dark']):

        self.Mask_Im_data={}
        self.Mask_Im_data['src_img'] = dataset
        for mtype in mtype_list:
            for i in np.arange(self.history):
                SWsweep,_ = self.SweepMask(dataset,ksize,stride,[i],mtype)
                SW2.append(SWsweep)
            Sw2=np.asarray(SW2)
            mse = np.square(Sw2[:,:,-1]-outdat*np.ones([Sw2.shape[0],Sw2.shape[1]]))

            Mask_Im_map = []
            for i in xrange(history):
                imshape=dataset.shape
                itno=0
                mask_map = np.zeros(imshape)
                for xind in np.arange(ksize,imshape[0],stride):
                    for yind in np.arange(ksize,imshape[1],stride):
                        mask=self.make_mask(Im,xind,yind,ksize)
                        mask = (1-mask)*1.0
                        mask_map = mask_map+mask*mse[i,itno]
                        itno = itno+1
                Mask_Im_map.append(mask_map/ksize*ksize*4)

            self.Mask_Im_data['mask_'+mtype] = np.asarray(Mask_Im_map)
        return self.Mask_Im_data
