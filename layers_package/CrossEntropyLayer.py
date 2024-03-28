import numpy as np

class CrossEntropy():
    def eval(self, Y, Yhat, epsilon=1e-12):
        Yhat = np.clip(Yhat, epsilon, 1. - epsilon)
        ce = -np.sum(Y*np.log(Yhat+1e-9))/Yhat.shape[0]
        return ce

    def gradient(self, Y, Yhat):
        return -1*(Y/(Yhat+10**-12))