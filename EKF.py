import numpy as np
import pylab as pl
from keras.layers import Input, Dense, Reshape, Multiply, MaxPool1D, Layer, BatchNormalization, Flatten, Conv2D, MaxPool3D,Dropout,GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress # For displaying training progress bar
from IPython.display import display, clear_output 
from keras.datasets import mnist
from keras.utils import to_categorical
from scipy.signal import convolve2d,convolve
from scipy.ndimage import convolve
import tensorflow as tf

class EKFCNN():

    def __init__(self,probes,weights):
        self.probes = probes
        self.weight_layers = np.array(weights)[[0,6,12,18,24]]
        self.sess = tf.Session()
        self.successive_layer = []

    def get_probe_layers(self):
        layers = [str(i) for i in range(1,6)]
        self.pred_layers = [self.model.probe_FCN(layer) for layer in layers]
        
    def prop_Q(self,x_ensemble):
        print('Getting Qs...')
        self.x_ensemble = x_ensemble
        self.pred_outputs = [pred.predict(self.x_ensemble) for pred in self.probes]
        self.Qs = [get_QCNN(pred_out) for pred_out in self.pred_outputs]
        print([np.mean(q) for q in self.Qs])
        print('Q matrices computed.')

    def pred_uncertainty(self,x):
        # initialize a tensorflow session
        sess = self.sess
        Qs = np.copy(self.Qs)
        weight_layers = self.weight_layers#[model_layer.get_weights()[0] for model_layer in self.model] #this may not be correct
        #model_covar_matrix = np.zeros((30,30))
        diffX = np.zeros_like(x)
#        diffX += x
        for i in range(30):
            if i != 29:
                diffX[:,:,i] = x[:,:,i+1] - x[:,:,i]
            else:
                diffX[:,:,i] = x[:,:,i] - x[:,:,i-1]
        covmat_0 = np.array([np.std(diffX[:,:,i])*np.std(diffX[:,:,j]) for i in range(30) for j in range(30)]).reshape(30,30)
#        print(covmat_0)
#        stds_diag = [np.std(diffX[:,:,i])**2 for i in range(30)]
#        print('Initial cov diagonals: ',stds_diag)
#        covmat_0 = np.diag(stds_diag)
#        covmat_0 = np.identity(30)
#        print('Initial covariance estimate: {}'.format(stds_diag))
        pred_layers_b = [x]
        [pred_layers_b.append(pred.predict(x)) for pred in self.probes]
        offset_pred_layers = [[x,l1,l2] for i,(l1,l2) in zip(range(5),zip(pred_layers_b[:6],pred_layers_b[1:]))]
#        print('Number of predictive layers: ',len(pred_layers_b))
#        print('Number of probes: ',len(self.probes))
#        pred_layers_b = [pred.predict(x) for pred in self.probes]
#        print('Total number of probes is: ',len(pred_layers_b))
        jacobians = [jacobianCNN(plb,W,probe,output=False) for i,(W,(probe,plb)) in enumerate(zip(weight_layers,(zip(self.probes,offset_pred_layers))))]
        print('Shape of all Jacobians',np.shape(jacobians))
        # plb is now layer l-1 and layer l
        covar_out = covar_recursive(covmat_0,jacobians,Qs)
        print('covar_out',np.shape(covar_out))
        return np.sqrt(np.diagonal(covar_out))

    def run_EKF(self,x_ensemble):
        self.prop_Q(np.array(x_ensemble))
    
def covar_recursive(cov0,jacobians,Qs):
#    print('Cov shape: {}'.format(np.shape(cov0)))
#    print('Jacobian shape: {}'.format(np.shape(jacobians[0])))
    if len(jacobians) == 0:
        return cov0
    else:
        jlen = len(jacobians[0])
        #jacob = jacobians[0][0]
        jacobs_masked = np.ma.array(jacobians[0],mask=jacobians[0]<0.)
        jacob = np.ma.mean(jacobians[0],axis=0)
        sigmas = (np.matmul(jacob,np.matmul(cov0,jacob.T)))
#        sigmas = np.nanmean([(np.matmul(jacob,np.matmul(cov0,jacob.T))) for jacob in jacobians[0]],axis=0)
#        sigmas = (np.matmul(jacob,np.matmul(cov0,jacob.T)))
#        sigmas = np.sqrt(np.nansum([(np.matmul(jacob,np.matmul(cov0,jacob.T)))**2 for jacob in jacobians[0]],axis=0))
        sigmaQ = sigmas + Qs[0]
#        sigmaQ /= Qsh[0]*Qsh[1]
#        sigmaQ /= np.nanstd(sigmaQ)
        #sigmaQ -= np.nanmean(sigmaQ)
#        sigmaQ /= np.nansum(sigmaQ)#np.nansum(sigmaQ)
#        sigmaQ = np.where(sigmaQ>0.,sigmaQ,0.)
#        sigmas /= np.nanstd(sigmas)
#        A = np.matmul(cov0,jacobians[0].T)
#        C = np.matmul(jacobians[0],A)
#        print('C: ',np.shape(C))
        print(np.mean(cov0))
        return covar_recursive(sigmaQ,jacobians[1:],Qs[1:])
#       return covar_recursive(np.matmul(jacobians[0],np.matmul(cov0,jacobians[0].T))+Qs[0],jacobians[1:],Qs[1:])

def jacobianCNN(x,W,probe,output=False):
    #x = x.squeeze()
    x0 = x[0].squeeze()
    xlminus1 = x[1].squeeze()
    xl = x[2].squeeze()

    print('Initial W shape ',np.shape(W))
    print('x_l-1 shape',np.shape(xlminus1))
    print('xl',np.shape(xl))
#    xl -= np.mean(xl)
#    xl /= np.std(xl)
#    if np.ndim(xl)>1:
#        W_conv = np.copy(W)
#        W_conv = probe.predict(np.expand_dims(np.ones_like(x0),axis=0))
    #W_conv = jacob_tensor(W,xl)
    W_conv = tensor_conv2d(W,xl)
#    else:
#        W_conv = np.copy(W)
        #W_conv = tensor_conv2d(W,xl)
#    W_conv /= np.max(W_conv)
#    W_conv *= np.max(W)
#    W_conv = np.copy(W)
#    W_conv_std = np.nan_to_num(np.ma.std(np.ma.array(W_conv,mask=W_conv<0.),axis=(0,1)))
#    print('W_conv_std: ',W_conv_std)

#    W_conv /= W_conv_std
#    W_conv = np.nan_to_num(W_conv)
#    W_std = np.nan_to_num(np.ma.std(np.ma.array(W,mask=W<0.),axis=(0,1)))
#    print('W_std: ',W_std)
#    W_conv *= W_std
#    W_conv = np.where(W_conv > 0.,W_conv,0.)
    #
#    Wsh = np.shape(W)
##    W_conv /= Wsh[0]*Wsh[1]
#    W_conv /= np.std(W_conv)
#    W_conv *= np.std(W)

    Wsh = np.shape(W_conv)
    print('Shape of new W conv',np.shape(W_conv))
#    W_conv = np.where(W_conv>0.,W_conv,0.)
    xl_sh = np.shape(xl)
    print('xl shape:',xl_sh)
#    if np.ndim(xl)>1:
#        #x_ = np.average(xl,axis=(0,1),weights=(np.abs(xl)+(1e-5)*np.random.randn(*np.shape(xl))))
#        x_ = xl.mean(axis=(0,1))
#    else:
#        x_ = np.copy(xl)
#    print(x_)
#    try:
#        x_ = [-1 if len(np.argwhere(xl[:,:,i]<0))/(1.*np.size(xl[:,:,i])) >= .1 else 1 for i in range(xl_sh[-1])]
#    except:
#        x_ = [-1 if len(np.argwhere(xl[i]<0))/(1.*np.size(xl[i])) >= .1 else 1 for i in range(xl_sh[-1])]
    # average over filters

#    W_conv = W_conv.squeeze()
    W_conv = W_conv.reshape(-1,Wsh[2],Wsh[3])
    print('Wconv shape',np.shape(W_conv))
    Wsh = np.shape(W_conv)
#    for i in range(xl_sh[-1]):
#        W_conv[:,:,:,i] = np.where(xl[:,:,i] > 0.,W_conv[:,:,:,i],0.)
#    if np.ndim(W_conv) > 2:
#        W_conv_ = np.ma.array(W_conv,mask=W_conv<0.)
#        W_conv = np.asarray(np.ma.mean(W_conv_,axis=(0,1)))
#        W_conv = np.average(W_conv_,axis=(0,1),weights=(W_conv + (1e-8)*np.random.randn(*np.shape(W_conv))))#np.average(W_conv,axis=(0,1))
#    W_conv = np.where(W_conv<0.,0.,W_conv)
    W_conv = np.nan_to_num(W_conv)
    F = np.zeros((Wsh[0],Wsh[2],Wsh[1]))
#    F = np.zeros((Wsh[1],Wsh[0]))
    print('F shape',np.shape(F))
    for i in range(Wsh[-1]):
 #       if x_[i] > 0.:
        F[:,i,:] = W_conv[:,:,i]#np.where(W_conv[:,i]<0.,0.,W_conv[:,i])#W[:,:,:,i]
    print('F shape:',np.shape(F))
    return F

def relu_where(W,x):
    sh = np.shape(W)
    W_out = np.zeros_like(W)
    for i in range(sh[3]):
        for j in range(sh[2]):
            W_out[:,:,j,i] = np.where(x[:,:,i] > 0.,W[:,:,j,i],0.)
    return W_out
            
def get_QCNN(x):
    def Q_(x_pred):
        # N x filter out
        x_sh = np.shape(x_pred)
        Q = np.zeros((x_sh[-1],x_sh[-1]))
        x_mu = np.mean(x_pred,axis=0)
        for i in range(x_sh[-1]):
            for j in range(x_sh[-1]):
                Q[j,i] = np.sum((x_pred[:,i]-x_mu[i])*(x_pred[:,j]-x_mu[j]))
        return Q/(x_sh[0]-1)
    x_sh = np.shape(x)
    print('shape of x ',np.shape(x))
    # N x spatial1 x spatial2 x filter
#    x = np.where(x>0.,x,0.)
    try:
        x = x.reshape(x_sh[0],-1,x_sh[3])
        Q_quad = np.array([Q_(x[:,i,:]) for i in range(np.shape(x)[1])])
#        Q_quad = np.sqrt(np.sum([Q_(x[:,i,:])**2 for i in range(np.shape(x)[1])],axis=0))
#        Q_quad /= np.nansum(Q_quad)
    except:
        Q_quad = np.array(Q_(x))
    Q_quad_mask = np.ma.array(Q_quad,mask=Q_quad<=0.)
    Q_quad = np.ma.median(Q_quad_mask,axis=0)
#    x_sh = np.shape(x)
#    print('x shape',np.shape(x))
#    Q = np.zeros((x_sh[-1],x_sh[-1]))
#    if np.ndim(x) > 2:
#        #samples x spatial1 x spatial2 x filters
#        x_red = np.mean(x,axis=(1,2))
#        x_mu = x_red.mean(axis=0) # average over all samples
#    else:
#    x_mu = x.mean(axis=0)
    # x batchx512x512x20  --> batchx20
#    for i in range(x_sh[-1]):
#        for j in range(x_sh[-1]):
#            try:
#                Q[i,j] = np.sum((x_red[:,i]-x_mu[i])*(x_red[:,j]-x_mu[j]))
#            except:
#                Q[i,j] = np.sum((x[:,i]-x_mu[i])*(x[:,j]-x_mu[j]))
#    Q_quad = np.zeros((x_sh[-1],x_sh[-1]))
    return Q_quad#Q/(x_sh[0]-1.)


def jacob_tensor(W,X):
    Wsh = np.shape(W)
    xsh = np.shape(X)
    print('JT W',Wsh)
    print('JT X',xsh)
    F = np.zeros_like(W)
    outW = np.zeros_like(W)
    X_ = np.ones_like(X)
#    X_ = np.where(X>0.,X_,0.)
    
    WXNew = [convolve2d(W[:,:,i,j],X[:,:,i],mode='same') for i in range(Wsh[2]) for j in range(Wsh[3])]
    WNew = [convolve2d(W[:,:,i,j],X_[:,:,i],mode='same') for i in range(Wsh[2]) for j in range(Wsh[3])]
#    WNew /= np.std(WNew,axis=(0,1))
#    WNew *= np.std(W,axis=(0,1))
#    if Wsh[2] > xsh[2]:
#        return W.mean(axis=(0,1))
    #for kx in range(Wsh[0]):
    #    for ky in range(Wsh[1]):
#            for kx in range(Wsh[0]):
#                for ky in range(Wsh[1]):
#    WX = np.where(WXNew > 0,WNew,0.)
#    WX = np.where(WX > 0.,WX,0.)
    WXmasked = np.ma.array(WX,mask=WX<0)
    WXmean = np.ma.mean(WXmasked,axis=(0,1))
#    WXmean /= np.max(WXmean)
#    WXmean *= np.max(W)
    return WXmean

def tensor_conv2d(W,X):
#    if np.ndim(xsh) == 1:
#        xsh = np.shape(W)
    sh = np.shape(W)
    shx = np.shape(X)
#    X_ = np.where(X>0.,1.,0.)
    try:
        X_ = np.ones_like(X)
#        X_ = np.where(X>0.,X_,0.)
    except:
        X_ = np.ones(sh[0])
#    X_ = np.where(X<0.,0.,1.)
#    out_ = np.zeros_like(W)#np.zeros((xsh[0],xsh[1],sh[2],sh[3]))
    #print('Tensor Conv W shape',np.shape(W))
    #print('Tensor Conv X_ shape',np.shape(X_))
#    out = np.copy(W)
    out = np.array([convolve2d(X_[:,:,i],W[:,:,j,i],mode='same') for i in range(sh[3]) for j in range(sh[2])]).reshape(shx[0],shx[1],sh[2],sh[3])
    print('X_ sh',np.shape(X_))
    print('OUT',np.shape(out))
#    out = np.copy(W)
    #out = np.reshape(out,(shx[0],shx[1],sh[2],sh[3]))
#    out = np.reshape(out,(sh[0],sh[1],sh[2],sh[3]))
#    out = np.where(out>0.,out,0.)
#    try:
#        X_L = np.array([convolve2d(X_,X[:,:,i],mode='same') for i in range(shx[-1])]).reshape(sh[0],sh[1],shx[-1])
#    except:
#        X_L = np.array([convolve(X_,X[i],mode='same') for i in range(shx[-1])]).reshape(sh[0],sh[1],shx[-1])
    out_ = np.zeros_like(out)
    for i in range(sh[3]):
        for j in range(sh[2]):
            if j==i:
                out_[:,:,j,i] = np.where(X_[:,:,i]>0.,out[:,:,j,i],0.)#out[:,:,j,i],0.)#convolve2d(W[:,:,j,i],X_[:,:,j],mode='same')
#            else:
#                out[:,:,j,i] = 0.
#    out = np.where(out>0.,out,0.)
#    out /= sh[2]*sh[3]
#    outma = np.ma.array(out,mask=out<=0.)
#    outmean = np.ma.mean(outma,axis=(0,1))
    return out_

def max_pool(cube,rate=2):
    N = np.shape(cube)
    #avg_rate = int(sample_rate/(300./N[0]))
    if N[0]%rate != 0.:
        cube = cube[:-1,:-1,:,:]
        cube_ = cube.reshape((N[0]-1)/rate,rate,(N[1]-1)/rate,rate,N[2],N[3])
    else:
        cube_ = cube.reshape(N[0]/rate,rate,N[1]/rate,rate,N[2],N[3])
    pooled_cube = cube_.max(axis=(1,3))
    return pooled_cube
