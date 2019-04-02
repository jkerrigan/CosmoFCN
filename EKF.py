import numpy as np
import pylab as pl
from keras.layers import Input, Dense, Reshape, Multiply, MaxPool1D, Layer, BatchNormalization, Flatten, Conv2D, MaxPool2D,Dropout,GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress # For displaying training progress bar
from IPython.display import display, clear_output 
from keras.datasets import mnist
from keras.utils import to_categorical


class EKFCNN():

    def __init__(self,model):
        self.model = model

    def get_probe_layers(self):
        layers = [str(i) for i in range(5)]
        self.pred_layers = [self.model.probe_FCN(layer) for layer in layers]
        
    def prop_Q(self,x_ensemble):
        print('Getting Qs...')
        self.x_ensemble = x_ensemble
        self.pred_outputs = [pred.predict(self.x_ensemble) for pred in self.pred_layers]
        self.Qs = [get_QCNN(pred_out) for pred_out in self.pred_outputs]
        print('Q matrices computed.')

    def pred_uncertainty(self,x):
        Qs = np.copy(self.Qs)
        weight_layers = [model_layer.get_weights()[0] for model_layer in self.model] #this may not be correct
        covmat_0 = np.identity(1)
        pred_layers_b = [pred.predict(self.x_ensemble) for pred in self.pred_layers]
        jacobians = [jacobianCNN(plb,wl) for plb,wl in zip(pred_layers_b,weight_layers)]
        covar_out = covar_recursive(covmat_0,jacobians,Qs)
        return np.diagonal(covar_out)
        
    def covar_recursive(cov0,jacobians,Qs):
        if len(jacobians) == 0:
            return cov0
        else:
            return covar_recursive(np.matmul(jacobians[0],np.matmul(cov0,jacobians[0]))+Qs[0],jacobians[1:],Qs[1:])

    def jacobianCNN(x,W):
        w_sh = np.shape(W)
        F = np.zeros((w_sh[2],w_sh[3]))
        if np.ndim(x) > 2:
            x = x.max(axis=(0,1,2))
        x = x.reshape(-1)
        W = W.max(axis=(0,1))
        for i in range(w_sh[-1]):
            if x[i]>0.:
                F[:,i] = W[:,i]
            return F.T
        
    def get_QCNN(x):
        x_sh = np.shape(x)
        print('x shape',np.shape(x))
        Q = np.zeros((x_sh[-1],x_sh[-1]))
        if np.ndim(x) > 2:
            x_red = x.max(axis=(1,2))
            x_mu = x.max(axis=(1,2)) # average over all samples
            x_mu = x_mu.mean(axis=0)
        else:
            x_mu = x.mean(axis=0)
        for i in range(x_sh[-1]):
            for j in range(x_sh[-1]):
                try:
                    Q[i,j] = np.sum((x_red[:,i]-x_mu[i])*(x_red[:,j]-x_mu[j]))
                except:
                    Q[i,j] = np.sum((x[:,i]-x_mu[i])*(x[:,j]-x_mu[j]))
        return Q/(x_sh[0]-1.)
