import numpy as np
import pylab as pl
from keras.layers import Input, Dense, Reshape, Multiply, MaxPool1D, Layer, BatchNormalization, Flatten, Conv2D, MaxPool3D,Dropout,GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from scipy.signal import convolve2d,convolve
from scipy.ndimage import convolve
import tensorflow as tf

class EKFCNN():
    '''
    Class for handling the propogation of uncertainties using the 
    Extended Kalman Filtering (EKF) technique. Accepts the weight initialized
    neural network probe layers along with the raw weights.
    '''

    def __init__(self,probes,weights,jacobian_mode=True):
        self.probes = probes # Probes are just the per layer conv. layer outputs
        self.jacobian_mode = jacobian_mode # if True it runs the dense approx
                                          # False then brute force jacobian

        # Picking the appropriate weights for the convolutional layers
        # Note: Doing it like this might not be necessary anymore
        self.weight_layers = np.array(weights)[[0,2,4,6,8]]
        
        self.sess = tf.Session() # Initialize tensorflow session
        
    def prop_Q(self,x_ensemble):
        '''
        Function for computing the sample covariance Q matrix by iteratively
        predicting through the probe layers.
        '''
        print('Getting Qs...')
        self.x_ensemble = x_ensemble
        self.pred_outputs = [pred.predict(self.x_ensemble) for pred in self.probes]
        self.Qs = [get_QCNN(pred_out) for pred_out in self.pred_outputs]
        print('Q matrices computed.')

    def pred_uncertainty(self,x):
        '''
        Function for predicting the uncertainty of an observation, by
        propogating the estimation of the model and observation noise.
        '''
        sess = self.sess
        Qs = np.copy(self.Qs)
        weight_layers = self.weight_layers
        print('Initial variance estimate of entire cube: {0}'.format(np.var(x)))
        
        mat_0 = np.matmul(np.sqrt(np.sum(x**2,axis=(0,1,2))).reshape(30,1),np.ones((1,30)))
        covmat_0 = np.cov(mat_0)
#covmat_0 = np.array([np.std(x[:,:,i])*np.std(x[:,:,j]) for i in range(30) for j in range(30)]).reshape(30,30)
        pred_layers_b = [x]
        [pred_layers_b.append(pred.predict(x)) for pred in self.probes]

        if self.jacobian_mode:
        # For dense EKF approximation
            jacobians = [DenseApproxJacob(pred_layers_b[i],pred_layers_b[i+1],weight_layers[i]) for i in range(len(pred_layers_b)-1)]

        else:
        # For brute force EKF (extremely slow!!!!)        
            jacobians = [jacobian_brute(pred_layers_b[i],pred_layers_b[i+1],weight_layers[i]) for i in range(len(pred_layers_b)-1)]

        covar_out = covar_recursive(covmat_0,jacobians,Qs,self.jacobian_mode)
        return np.sqrt(np.diagonal(covar_out))

    def run_EKF(self,x_ensemble):
        self.prop_Q(np.array(x_ensemble))
    
def covar_recursive(cov0,jacobians,Qs,jacobian_mode=True):
#    print('Cov shape: {}'.format(np.shape(cov0)))
#    print('Jacobian shape: {}'.format(np.shape(jacobians[0])))
    if len(jacobians) == 0:
        return cov0
    else:
        jlen = len(jacobians[0])
        jacob = jacobians[0]

    if jacobian_mode:
        # For dense approx. jacobians
        sigmas = (np.matmul(jacob,np.matmul(cov0,jacob.T)))
    else:
        # For brute force jacobians
        sig = np.matmul(cov0,np.mean(jacob.T,axis=(2,3)))
#        sig = np.einsum('ij,jklm->iklm',cov0,jacob.T)
        sigmas = np.matmul(np.mean(jacob**2,axis=(0,1)),sig)
#        sigmas = np.einsum('ijkl,lmji->km',jacob,sig)

    sigmaQ = sigmas + Qs[0]
    return covar_recursive(sigmaQ,jacobians[1:],Qs[1:],jacobian_mode=jacobian_mode)


def DenseApproxJacob(A,B,W):
    '''
    This method for computing the jacobian averages over
    the spatial dimensions to treat each convolutional layer's 
    filter weights as if they were just a dense layer.
    There still is more work required to show if this is
    a correct approximation.
    '''
    # A is the prior (l-1) layer prediction
    # B is the current (l) layer prediction
    Ash = np.shape(A)
    Bsh = np.shape(B)
    Wsh = np.shape(W)
    try:
        # Average over all spatial dimensions
        # Note: First dimension is batch
        A_ = np.mean(A,axis=(0,1,2))
        B_ = np.mean(B,axis=(0,1,2))
        W_ = np.mean(W,axis=(0,1))
    except:
        # Special case of spatial averaging due to global maxpooling
        A_ = np.mean(A,axis=(0,1,2))
        B_ = np.mean(B,axis=0)
        W_ = np.mean(W,axis=(0,1))

    F = np.zeros((Wsh[3],Wsh[2]))
    for i in range(len(A_)):
        for j in range(len(B_)):
            if B_[j] > 0.:
                F[j,i] = W_[i,j]
    return F
            
def get_QCNN(x):

    '''
    Method for calculating the per layer output sample covariance matrix.
    Currently reduces the spatial dimensions by averaging.
    '''

    def Q_(x_pred):
        # Computes the sample covariance matrix Q
        # Input shape should be (Batch x spatial dims reshaped x filters)
        x_sh = np.shape(x_pred)
        Q = np.zeros((x_sh[-1],x_sh[-1]))
        x_mu = np.mean(x_pred,axis=0)
        for i in range(x_sh[-1]):
            for j in range(x_sh[-1]):
                Q[j,i] = np.sum((x_pred[:,i]-x_mu[i])*(x_pred[:,j]-x_mu[j]))
        return Q/(x_sh[0]-1)
    x_sh = np.shape(x)
    try:
        x = x.reshape(x_sh[0],-1,x_sh[3])
        Q_quad = np.array([Q_(x[:,i,:]) for i in range(np.shape(x)[1])])
    except:
        Q_quad = np.array(Q_(x))
    # Add all of the spatial components in quadrature
    Q_quad = np.mean(Q_quad,axis=0)
#    Q_quad = np.zeros_like(Q_quad)
    return Q_quad

def jacobian_brute(x0,x1,W):
    '''
    The brute force method of computing the Jacobian of a 2d convolutional
    layer. Chock full of nested for loops, this guy takes a considerable amount
    of time to run. See Paul's derivation in the CNN-EKF Memo.
    '''
    x0 = x0.squeeze()
    x1 = x1.squeeze()
    try:
        kx_,ky_,j_ = np.shape(x1)
    except:
        kx_ = 1
        ky_ = 1
        j_ = len(x1)
    null1,null2,j_ = np.shape(x0)
    nx_,ny_,i_,j_ = np.shape(W)
    F = np.zeros((kx_,ky_,j_,i_)) 
    for i in range(i_):
        for j in range(j_):
            for kx in range(kx_):
                for ky in range(ky_):
                    for nx in range(-nx_/2 + 1,nx_/2 + 1):
                        for ny in range(-ny_/2 + 1,ny_/2 + 1):
                            try:
                                if np.ndim(x1) > 2:
                                    if x1[kx,ky,j]>0:
                                        F[kx,ky,j,i] += W[kx-nx,ky-ny,i,j]
                                else:
                                    if x1[j]>0:
                                        F[kx,ky,j,i] += W[kx-nx,ky-ny,i,j]
                            except IndexError:
                                pass
    return F
