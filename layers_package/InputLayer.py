import numpy as np
from .Layer import Layer

class InputLayer(Layer):
    def __init__(self, dataIn):
        self.__prevIn = (dataIn)
        self.__prevOut =([])

    def forward(self, dataIn):
        data = np.array(dataIn)
        zscoreData = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)
        zscoreData[np.isnan(zscoreData)] = 0
        self.__prevOut = zscoreData
        return zscoreData

    def gradient(self):
        pass
    def backward(self, gradIn):
        pass

