import numpy as np
from .Layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.__prevIn = dataIn
        flattened = []
        for i in range(dataIn.shape[0]):
            img_flat = dataIn[i, :, :, :].flatten()
            flattened.append(img_flat)

        flattened = np.atleast_2d(flattened)
        self.__prevOut = flattened
        return flattened
    
    def gradient(self, gradIn):
        unflattened = []
        for i in range(gradIn.shape[0]):
            unflat_img = gradIn[i,:].reshape(self.__prevIn.shape[1:])
            unflattened.append(unflat_img)
        unflattened = np.atleast_2d(unflattened)
        return unflattened

    def backward(self, gradIn):
        return self.gradient(gradIn)