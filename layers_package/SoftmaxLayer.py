import numpy as np
from .Layer import Layer

class SoftMaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        data = np.array(dataIn)
        maxes = np.max(data, axis=1, keepdims=True)
        data -= maxes
        self.setPrevIn(dataIn)
        softMaxAct = np.exp(dataIn)/(np.sum(np.exp(dataIn),axis=1, keepdims=True) + 10**7)
        self.__prevOut = softMaxAct
        return softMaxAct

    def gradient(self):
        sm_tensor = np.array([np.diag(row) - np.matmul(np.atleast_2d((row)).T, np.atleast_2d((row))) for row in self.__prevOut])
        return sm_tensor
        
    def backward(self, gradIn):
        return np.einsum('...i,...ij', gradIn, self.gradient())

