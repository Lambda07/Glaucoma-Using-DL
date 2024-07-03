import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import random
import math
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras.utils import plot_model
from tabulate import tabulate
from scipy import ndimage
matplotlib.rcParams['figure.figsize']=(20.0,10.0)

def FFN(inputDim,outputDim):
    ffn=Sequential()
    ffn.add(Dense(1024,input_dim=inputDim,init='uniform',activation='tanh'))
    ffn.add(Dense(1024,input_dim=1024,init='uniform',activation='tanh'))
    ffn.add(Dense(outputDim,input_dim=1024,init='uniform'))
    ffn.add(Activation('sigmoid'))
    ffn.summary()
    return ffn

def downSample(flatPicture,factor):
    newImage=np.zeros(len(flatPicture)//factor)
    i=0;
    for iPixel in range(len(flatPicture)):
        if(iPixel % factor==0) & (i<len(newImage)):
            newImage[i]=flatPicture[iPixel]
            i+=1
    return newImage

def TrainNetwork(compiledNetwork,trainX,trainY,epochs):
    for iEpoch in range(epochs):
        compiledNetwork.train_on_batch(trainX,trainY)
        if iEpoch%100==0:
            print(iEpoch)
    return compiledNetwork

originalGlaucomaTrain=[None]*45
trainX=np.zeros((100,240))
for i in range(45):
    if i<9:
        iPic='0'+str(i+1);
    else:
        iPic=''+str(i+1);
    face=ndimage.imread('F:/glaucoma/g_'+iPic+'.jpg',flatten=True)
    print('F:/glaucoma/g_'+iPic+'.jpg')
    originalGlaucomaTrain[i]=face;
    face=face.flatten()
    trainX[i,:]=downSample(face,1000)/256.0

originalHealthyTrain=[None]*45
for i in range(45):
    if i<9:
        iPic='0'+str(i+1);
    else:
        iPic=''+str(i+1);
    face=ndimage.imread('F:/healthy/h_'+iPic+'.jpg',flatten=True)
    print('F:/healthy/h_'+iPic+'.jpg')
    originalHealthyTrain[i]=face;
    face=face.flatten()
    trainX[i+46,:]=downSample(face,1000)/256.0

trainY=np.zeros((100,1))
for i in range(45):
    trainY[i]=1
    
net=FFN(trainX.shape[1],1)

net.compile(loss='binary_crossentropy',optimizer="SGD")


trainedNet=TrainNetwork(net,trainX,trainY,2000)

for iPic in range(45):
    f,axarr=plt.subplots(1,2)
    axarr[0].imshow(originalGlaucomaTrain[iPic],cmap='Greys_r')
    axarr[0].set_title('Glaucoma Train')
    axarr[0].get_xaxis().set_visible(True)
    axarr[0].get_yaxis().set_visible(True)
    axarr[1].imshow(originalHealthyTrain[iPic],cmap='Greys_r')
    axarr[1].set_title('Healthy Train')
    axarr[1].get_xaxis().set_visible(False)
    axarr[1].get_yaxis().set_visible(False)


testX=np.zeros((10,240))
originalGlaucomaTest=[None]*5
originalHealthyTest=[None]*5
i=1
for i in range(5):
    if i<9:
        iPic=''+str(i+46);
    else:
        iPic=''+str(i+46);
        
    face=ndimage.imread('F:/glaucoma/g_'+iPic+'.jpg',flatten=True)
    print('F:/glaucoma/g_'+iPic+'.jpg')
    originalGlaucomaTest[i]=face;
    face=face.flatten()
    testX[i,:]=downSample(face,1000)/256.0
for i in range(5):
    if i<9:
        iPic=''+str(i+46);
    else:
        iPic=''+str(i+46);
    face=ndimage.imread('F:/healthy/h_'+iPic+'.jpg',flatten=True)
    originalHealthyTest[i]=face;
    face=face.flatten()
    testX[i+5,:]=downSample(face,1000)/256.0
    print('F:/healthy/h_'+iPic+'.jpg')

out=trainedNet.predict(testX)
print(out)
correctTestY=[True,True,True,True,True,False,False,False,False,False]
for iPic in range(5):
    f,axarr=plt.subplots(1,2)
    print(f)
    axarr[0].imshow(originalGlaucomaTest[iPic],cmap='Greys_r')
    axarr[0].set_title('Glaucoma Test ID'+str(out[iPic]>.5))
    print(out[iPic])
    axarr[0].get_xaxis().set_visible(True)
    axarr[0].get_yaxis().set_visible(True)
    axarr[1].imshow(originalHealthyTest[iPic],cmap='Greys_r')
    axarr[1].set_title('Healthy Test ID'+str(out[iPic+5]>.5))
    print(out[iPic])
    axarr[1].get_xaxis().set_visible(False)
    axarr[1].get_yaxis().set_visible(False)
    
def identification(pred,act):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(act)):
        if act[i]:
            if pred[i]:
                TP+=1.0
            else:
                FN+=1.0
        else:
            if pred[i]:
                FP+=1.0
            else:
                TN+=1.0
        return TP,TN,FP,FN
idTable=np.asarray([range(10),correctTestY,(out*100).flatten(),(out>.5).flatten()]).T.tolist()
print(tabulate(idTable,headers=['Picture #','Glaucoma Actual','Glaucoma Percentage','Glaucoma Predicted']))
TP,TN,FP,FN=identification((out>.5).flatten(),correctTestY)
print(TP,TN,FP,FN)

