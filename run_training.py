
from layers_package.InputLayer import InputLayer
from layers_package.SoftmaxLayer import SoftMaxLayer
from layers_package.FullyConnectedLayer import FullyConnectedLayer
from layers_package.CrossEntropyLayer import CrossEntropy
from layers_package.ConvolutionLayer import ConvolutionLayer
from layers_package.MaxPoolLayer import MaxPoolLayer
from layers_package.FlattenLayer import FlattenLayer
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def accuracy(y,yhat):
    tp = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            tp+=1

    return tp/len(y)

def onehot(Y_colv):
    ohe = np.zeros((len(Y_colv), len(np.unique(Y_colv))))
    for i in range(len(Y_colv)):
        encoding = Y_colv[i][0]
        ohe[i][encoding] = 1
    return ohe


key_train = pd.read_csv('breeds.csv')

Xtrain = []
Ytrain = []


for index, row in key_train.iterrows():

    path= os.path.join('img',f'{row[2]}',f'{row[3]}')

    Xtrain.append(np.array(Image.open(path).resize((40,40))))
    Ytrain.append(row[1])

Xtrain = np.array(Xtrain)/255
Ytrain = np.atleast_2d(Ytrain).T

Yt_enc = onehot(Ytrain)

convLayer = ConvolutionLayer(3,3,5)
mpLayer = MaxPoolLayer(3,3)
flatLayer = FlattenLayer()
H = Xtrain

H = convLayer.forward(H)
H = mpLayer.forward(H)
H = flatLayer.forward(H)

FCLayer = FullyConnectedLayer(H.shape[1], Yt_enc.shape[1])
lsLayer = SoftMaxLayer()
llLayer = CrossEntropy()

layers = [convLayer, mpLayer, flatLayer, FCLayer, lsLayer, llLayer]

H = Xtrain
epoch = 1

ce_train = []
acc_train = []
while epoch < 100:
    H=Xtrain

    for i in range(len(layers)-1):
        H = layers[i].forward(H)

    ce_train.append(layers[-1].eval(Yt_enc, H))
    t_acc = accuracy(Ytrain.flatten(), np.argmax(H, axis = 1))
    print("epoch", epoch, "acc",t_acc, 'loss', ce_train[-1])

    acc_train.append(t_acc)

    #backwards propagation
    grad = layers[-1].gradient(Yt_enc,H) 
    for i in range(len(layers)-2,0,-1):    
        newgrad = layers[i].backward(grad)

        if (isinstance(layers[i], FullyConnectedLayer)):
            layers[i].updateWeightsADAM(grad,epoch, eta=.01)
        grad= newgrad

    layers[0].updateWeights(grad,eta = 0.001)
   
    epoch +=1

plt.plot(ce_train)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.savefig(os.path.join("out", "training_loss.png"))



