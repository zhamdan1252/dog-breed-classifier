import numpy as np
from .Layer import Layer

class ConvolutionLayer(Layer):
    def __init__(self, kernel_height, kernel_width, n_kernels):
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.kernels = []
        for i in range(n_kernels):
            self.kernels.append(np.random.uniform(-1e-4,1e-4, size=(kernel_height,kernel_width,3)))

    def forward(self, dataIn):
        data = np.atleast_3d(dataIn)
        self.__prevIn = data
        conv_out = []
        for n in range(data.shape[0]):
            imgin = data[n,:,:,:]
            feature_maps = []
            for k in self.kernels:
                feature_maps.append(self.crossCorrelate3D(imgin, k))
            feature_maps = np.array(feature_maps)
            feature_maps = np.transpose(feature_maps, (1,2,0))
            conv_out.append(feature_maps)
        conv_out = np.array(conv_out)
        self.__prevOut = conv_out
        return conv_out

    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass
    

    def updateWeights(self, gradIn, eta = 0.001):
        X = self.__prevIn
        new_kernels = []
        for p in range(gradIn.shape[3]):
            average_k = np.zeros_like(self.kernels[0])
            for i in range(X.shape[0]):
                newK = self.crossCorrelateUpdateWeights(X[i,:,:,:], gradIn[i,:,:,p])
                average_k += newK
            average_k = average_k/X.shape[0]
            new_kernels.append(average_k)
        
        for i in range(len(self.kernels)):
            self.kernels[i] -= eta*new_kernels[i]
    
    
    def crossCorrelate3D(self,matrix, kernel):
        fm_height = matrix.shape[0]-kernel.shape[0] + 1
        fm_width = matrix.shape[1]-kernel.shape[1] + 1
        feature_map = np.zeros((fm_height, fm_width))
        for i in range(fm_height):
            for j in range(fm_width):
                region = matrix[i:i+kernel.shape[0], j:j+kernel.shape[1], :]
                feature_map[i, j] = np.sum(region * kernel)
        return feature_map
    
    def crossCorrelateUpdateWeights(self, prevIn, gradIn):
        fm_height = prevIn.shape[0]-gradIn.shape[0] + 1
        fm_width = prevIn.shape[1]-gradIn.shape[1] + 1
        newK = np.zeros((fm_height, fm_width, 3))
        for d in range(3):
            for i in range(fm_height):
                for j in range(fm_width):
                    region = prevIn[i:i+gradIn.shape[0], j:j+gradIn.shape[1], d]
                    newK[i, j, d] = np.sum(region * gradIn)
        return newK
    
