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
from scipy.signal import convolve2d


class EKFCNN():

    def __init__(self,probes,weights):
        self.probes = probes
        self.weight_layers = np.array(weights)[[0,6,12,18,24]]

    def get_probe_layers(self):
        layers = [str(i) for i in range(5)]
        self.pred_layers = [self.model.probe_FCN(layer) for layer in layers]
        
    def prop_Q(self,x_ensemble):
        print('Getting Qs...')
        self.x_ensemble = x_ensemble
        self.pred_outputs = [pred.predict(self.x_ensemble) for pred in self.probes]
        self.Qs = [get_QCNN(pred_out) for pred_out in self.pred_outputs]
        print([np.mean(q) for q in self.Qs])
        print('Q matrices computed.')

    def pred_uncertainty(self,x):
        Qs = np.copy(self.Qs)
        weight_layers = self.weight_layers#[model_layer.get_weights()[0] for model_layer in self.model] #this may not be correct
        stds_diag = [np.std(x[:,:,i]) for i in range(20)]
        covmat_0 = np.diag(stds_diag)#np.identity(20)
        print('Initial covariance estimate: {}'.format(stds_diag))
        pred_layers_b = [pred.predict(x) for pred in self.probes]
        jacobians = [jacobianCNN(plb,wl) for plb,wl in zip(pred_layers_b,weight_layers)]
        covar_out = covar_recursive(covmat_0,jacobians,Qs)
        return np.sqrt(np.diagonal(covar_out))

    def run_EKF(self,x_ensemble):
        self.prop_Q(np.array(x_ensemble))
    
def covar_recursive(cov0,jacobians,Qs):
#    print('Cov shape: {}'.format(np.shape(cov0)))
#    print('Jacobian shape: {}'.format(np.shape(jacobians[0])))
    if len(jacobians) == 0:
        return cov0
    else:
        print(np.max(np.matmul(jacobians[0],np.matmul(cov0,jacobians[0].T))))
        return covar_recursive(np.matmul(jacobians[0],np.matmul(cov0,jacobians[0].T))+Qs[0],jacobians[1:],Qs[1:])

def jacobianCNN(x,W):
    w_sh = np.shape(W)
    print('jacobian W shape:',w_sh)
    F = np.zeros((w_sh[3],w_sh[2]))
    print('x shape',np.shape(x))
    if np.ndim(x) > 3:
        x = np.median(x,axis=(0,1,2))
    else:
        x = np.median(x,axis=0)
    x = x.reshape(-1)
#    print('x shape',np.shape(x))
    W = tensor_conv2d(W)
    W = np.median(W,axis=(0,1))
    print('Max W',np.max(W))
    for i in range(w_sh[-1]):
        if x[i] > 0.:
            F[i,:] = W[:,i]
    return F

def relu_where(W,x):
    sh = np.shape(W)
    W_out = np.zeros_like(W)
    for i in range(sh[3]):
        for j in range(sh[2]):
            W_out[:,:,j,i] = np.where(x[:,:,i] > 0.,W[:,:,j,i],0.)
    return W_out
            
def get_QCNN(x):
    x_sh = np.shape(x)
    print('x shape',np.shape(x))
    Q = np.zeros((x_sh[-1],x_sh[-1]))
    if np.ndim(x) > 2:
        x_red = np.median(x,axis=(1,2))#x.median(axis=(1,2))
        #x_mu = np.median(x_mu,axis=(1,2))
        x_mu = x_red.mean(axis=0) # average over all samples
        #x_mu = x_mu.mean(axis=0)
    else:
        x_mu = x.mean(axis=0)
    # x batchx512x512x20  --> batchx20
    for i in range(x_sh[-1]):
        for j in range(x_sh[-1]):
            try:
                Q[i,j] = np.sum((x_red[:,i]-x_mu[i])*(x_red[:,j]-x_mu[j]))
            except:
                Q[i,j] = np.sum((x[:,i]-x_mu[i])*(x[:,j]-x_mu[j]))
#    Q = np.zeros((x_sh[-1],x_sh[-1]))
    return Q/(x_sh[0]-1.)

def tensor_conv2d(W):
    x = np.ones_like(W[:,:,0,0])
    sh = np.shape(W)
    out = np.zeros_like(W)
    for i in range(sh[2]):
        for j in range(sh[3]):
            out[:,:,i,j] = convolve2d(W[:,:,i,j],x,mode='same')
    return out
