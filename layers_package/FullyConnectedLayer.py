import math
import numpy as np
from .Layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut, s_dw=0, r_dw=0, s_db=0, r_db=0):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        weightsrange = np.sqrt(6.0 / (sizeIn + sizeOut))
        self.weights = np.random.uniform(-weightsrange, weightsrange, size=(sizeIn, sizeOut))
        self.biases = np.random.uniform(-1e-4, 1e-4, size=(1, sizeOut))
        self.s_dw = s_dw
        self.r_dw = r_dw
        self.s_db = s_db
        self.r_db = r_db
    
    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights = np.atleast_2d(weights)
        
    def getBiases(self):
        return self.biases
    
    def setBiases(self, biases):
        self.biases = biases

    def forward(self, dataIn):
    
        data = np.atleast_2d(dataIn)
        self.__prevIn = data
        fullyConnected = data @ self.weights + self.biases
        self.__prevOut = fullyConnected
        return fullyConnected
        
    def gradient(self):
        grad = self.weights.T
        return grad

    def backward(self, gradIn):
        return np.atleast_2d(gradIn @ self.gradient())
    
    def updateWeights(self, gradIn, eta = 0.001):
        dJdb = np.sum(gradIn, axis = 0)

        dJdW = ((self.__prevIn).T @ np.atleast_2d(gradIn)) / gradIn.shape[0]

        self.weights -= eta*dJdW
        self.biases -= eta*dJdb    

    def updateWeightsADAM(self, gradIn, t, eta=0.001):
        dJdb = np.sum(gradIn, axis=0)
        dJdW = ((self.__prevIn).T @ gradIn) / gradIn.shape[0]

        delta = 1e-8
        beta1 = 0.9
        beta2 = 0.999

        self.s_dw = beta1 * self.s_dw + (1 - beta1) * dJdW
        self.r_dw = beta2 * self.r_dw + (1 - beta2) * (dJdW * dJdW)

        s_dw_hat = self.s_dw / (1 - beta1**t)
        r_dw_hat = self.r_dw / (1 - beta2**t)

        self.weights -= eta * (s_dw_hat / (np.sqrt(r_dw_hat) + delta))
        self.biases -= eta * (dJdb / (1 - beta1**t))


    