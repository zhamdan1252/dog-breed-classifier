
import math
import numpy as np
from .Layer import Layer

class MaxPoolLayer(Layer):
    
    def __init__(self, width, stride):
        self.stride = stride
        self.width = width
        self.max_info = []

    def forward(self, dataIn):
        self.__prevIn = dataIn
        mp_matrix = []
        self.max_info = []
        for n in range(dataIn.shape[0]):
            mp_1img = []
            for p in range(0,(dataIn.shape[3])):
                mp_matrix_1fm = []
                for i in range(0,dataIn.shape[1]-self.width+1,self.stride):
                    mp_row = []
                    for j in range(0,dataIn.shape[2]-self.width+1,self.stride):
                        region = np.atleast_3d(dataIn[n,i:i+self.width,j:j+self.width,p])
                        max_val = np.max(region)
                        mp_row.append(max_val)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        max_info= [n,max_idx[1]+i, max_idx[2]+j, p, max_val]  
                        self.max_info.append(max_info)
                    mp_matrix_1fm.append(mp_row)
                mp_1img.append(mp_matrix_1fm)
            mp_matrix.append(mp_1img)
        mp_matrix = np.array(mp_matrix)
        mp_matrix = np.transpose(mp_matrix, (0, 2, 3, 1))
        self.__prevOut = mp_matrix
        return mp_matrix

    def gradient(self, gradIn):
        grad = np.zeros_like(self.__prevIn)

        for d in range(gradIn.shape[0]):
            grad_vals = gradIn[d].flatten()
            for i in range(len(grad_vals)):
                idx = self.max_info[i]
                grad[idx[0],idx[1],idx[2], idx[3]] = grad_vals[i]
        return grad


    def backward(self,gradIn):
        return self.gradient(gradIn)

