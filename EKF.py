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
        weight_layers = self.weight_layers #[model_layer.get_weights()[0] for model_layer in self.model] #this may not be correct
        [print() for weights in weight_layers]
        #model_covar_matrix = np.zeros((30,30))
#        diffX = np.zeros_like(x)
#        diffX = np.copy(x)
#        diffX += x
#        for i in range(30):
#            if i != 29:
#                diffX[:,:,i] = x[:,:,i+1] - x[:,:,i]
#            else:
#                diffX[:,:,i] = x[:,:,i] - x[:,:,i-1]
#        covmat_0 = np.array([np.std(diffX[:,:,i])*np.std(diffX[:,:,j]) for i in range(30) for j in range(30)]).reshape(30,30)
#        print(covmat_0)
#        stds_diag = [np.std(diffX[:,:,i])**2 for i in range(30)]
#        print('Initial cov diagonals: ',stds_diag)
#        covmat_0 = np.diag(stds_diag)
        covmat_0 = np.identity(30)
#        print('Initial covariance estimate: {}'.format(stds_diag))
        pred_layers_b = []
        [pred_layers_b.append(pred.predict(x)) for pred in self.probes]
        offset_pred_layers = [[x,l1,l2] for i,(l1,l2) in zip(range(5),zip(pred_layers_b[:6],pred_layers_b[1:]))]
#        print('Number of predictive layers: ',len(pred_layers_b))
#        print('Number of probes: ',len(self.probes))
#        pred_layers_b = [pred.predict(x) for pred in self.probes]
#        print('Total number of probes is: ',len(pred_layers_b))
        jacobians = []
        x_next = pred_layers_b[1]
        for i in range(len(pred_layers_b)-1):
            #print('x shape: {0}'.format(np.shape(x)))
            print('predict x0 shape: {0}'.format(np.shape(pred_layers_b[i])))
            print('predict x1 shape: {0}'.format(np.shape(pred_layers_b[i+1])))
            print('weights shape: {0}'.format(np.shape(weight_layers[i])))
            F_ = jacobian_v2(pred_layers_b[i],pred_layers_b[i+1],weight_layers[i+1])
            #F_ = np.where(F_>0.,F_,0.)
        #    F_,XL_ = jacobianCNN(self.probes[i].predict(x),weight_layers[i],self.probes[i],output=False)
            jacobians.append(F_)
            #x_next = np.copy(XL_)

#        jacobians = [jacobianCNN(plb,W,probe,output=False) for i,(W,(probe,plb)) in enumerate(zip(weight_layers,(zip(self.probes,offset_pred_layers))))]
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
        print('Shape of jacobian: {}'.format(np.shape(jacobians[0])))
        print('Shape of cov: {}'.format(np.shape(cov0)))
        jacob = jacobians[0]
        #jacob = np.mean(jacobians[0][0],axis=0)
        #jacobs_masked = np.ma.array(jacobians[0],mask=jacobians[0]<0.)
        #jacob = np.ma.mean(jacobs_masked,axis=0)
#        np.einsum('ij,kjlm->iklm',cov0,jacob.T)

#        sig = np.tensordot(cov0,jacob,axes=1)
#        sigmas = np.tensordot(jacob.T,sig,axes=1)

        sig = np.einsum('ij,kjlm->iklm',cov0,jacob.T)
        sigmas = np.einsum('ijkl,kmji->lm',jacob,sig)
        #sigmas /= (1.*jlen/2.)**2
        #sigmas = np.ma.dot(jacob,np.ma.dot(cov0,jacob.T))#(np.matmul(jacob,np.matmul(cov0,jacob.T)))
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

def jacobian_v2(x0,x1,W):
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
    #W = np.where(W>0.,W,0.)
    F = np.zeros((kx_,ky_,i_,j_)) 
    for kx in range(kx_):
        for ky in range(ky_):
            for j in range(j_):
                for i in range(i_):
                    for nx in range(-nx_/2 + 1,nx_/2 + 1):
                        for ny in range(-ny_/2 + 1,ny_/2 + 1):
                            try:
                                if np.ndim(x1) > 2:
                                    if x1[kx,ky,j]>0:
                                        F[kx,ky,i,j] += W[kx-nx,ky-ny,i,j]
                                else:
                                    if x1[j]>0:
                                        F[kx,ky,i,j] += W[kx-nx,ky-ny,i,j]
                            except IndexError:
                                pass
    return F

def DenseApproxJacob(A,B,W):
    # A is the prior layer prediction
    # B is the current layer prediction
    Ash = np.shape(A)
    Bsh = np.shape(B)
    Wsh = np.shape(W)
    try:
        A1 = np.random.choice(range(Ash[1]))
        A2 = np.random.choice(range(Ash[2]))
        B1 = np.random.choice(range(Bsh[1]))
        B2 = np.random.choice(range(Bsh[2]))
        W1 = np.random.choice(range(Wsh[0]))
        W2 = np.random.choice(range(Wsh[1]))
        A_ = A[0,A1,A2,:]#np.mean(A,axis=(0,1,2))#A.mean(axis=(0,1,2))
        B_ = B[0,B1,B2,:]#np.mean(B,axis=(0,1,2))#B.mean(axis=(0,1,2))
        W_ = W[W1,W2,:,:]#np.mean(W,axis=(0,1))#W.mean(axis=(0,1))
    except:
        A1 = np.random.choice(range(Ash[1]))
        A2 = np.random.choice(range(Ash[2]))
        #B1 = np.random.choice(range(Bsh[1]))
        #B2 = np.random.choice(range(Bsh[2]))
        W1 = np.random.choice(range(Wsh[0]))
        W2 = np.random.choice(range(Wsh[1]))
        A_ = A[0,A1,A2,:]#np.mean(A,axis=(0,1,2))
        B_ = B[0,:]#np.mean(B,axis=0)
        W_ = W[W1,W2,:,:]#np.mean(W,axis=(0,1))
        
    print('A shape: {0}'.format(np.shape(A_)))
    print('B shape: {0}'.format(np.shape(B_)))
    print('W shape: {0}'.format(np.shape(W_)))
    F = np.zeros((len(A_),len(B_)))
    for i in range(len(A_)):
        for j in range(len(B_)):
            if B_[j] > 0.:
                F[i,j] = W_[i,j]
    return F.T

def FinDiffJacobian(A,B,W):
    def findiff(y):
        dy = np.zeros_like(y)
        for i in range(len(y)-1):
            dy[i] = y[i+1] - y[i]
        return dy
    # A is the prior layer prediction
    # B is the current layer prediction
    A = A.mean(axis=(0,1,2))
    B = B.mean(axis=(0,1,2))
    W = W.mean(axis=(0,1))
    print('A shape: {0}'.format(np.shape(A)))
    print('B shape: {0}'.format(np.shape(B)))
    print('W shape: {0}'.format(np.shape(W)))
    F = np.zeros((len(B),len(A)))
    dA = findiff(A)
    print('max dA {0} min dA {1}'.format(np.min(dA),np.max(dA)))
    dB = findiff(B)
    for i in range(len(A)):
        for j in range(len(B)):
            #print(dB[j]/dA[i])
            if B[j] > 0.:
                F[j,i] = W[i,j]
    return F

def jacobianCNN(x,W,probe,output=False):
    #x = x.squeeze()
#    x0 = x[0].squeeze()
#    xlminus1 = x[1].squeeze()
    xl = x.squeeze()

    print('Initial W shape ',np.shape(W))
#    print('x_l-1 shape',np.shape(xlminus1))
    print('xl',np.shape(xl))
#    xl -= np.mean(xl)
#    xl /= np.std(xl)
#    if np.ndim(xl)>1:
#        W_conv = np.copy(W)
#        W_conv = probe.predict(np.expand_dims(np.ones_like(x0),axis=0))
    #W_conv = jacob_tensor(W,xl)
    W_conv = tensor_conv2d(W,xl)
    XL_conv = np.copy(W_conv)
    Wsh = np.shape(W_conv)
    print('Shape of new W conv',np.shape(W_conv))
    xl_sh = np.shape(xl)
    print('xl shape:',xl_sh)

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
        #if x_[i] > 0.:
        F[:,i,:] = W_conv[:,:,i]#np.where(W_conv[:,i]<0.,0.,W_conv[:,i])#W[:,:,:,i]
    print('F shape:',np.shape(F))
    return F,XL_conv

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
#    Q_quad_mask = np.ma.array(Q_quad,mask=Q_quad<=0.)
    Q_quad = np.mean(Q_quad,axis=0)
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
#    WXmean = np.ma.mean(WXmasked,axis=(0,1))
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
 #   print('OUT',np.shape(out))
#    out = np.copy(W)
    #out = np.reshape(out,(shx[0],shx[1],sh[2],sh[3]))
#    out = np.reshape(out,(sh[0],sh[1],sh[2],sh[3]))
#    out = np.where(out>0.,out,0.)
#    try:
#        X_L = np.array([convolve2d(X_,X[:,:,i],mode='same') for i in range(shx[-1])]).reshape(sh[0],sh[1],shx[-1])
#    except:
#        X_L = np.array([convolve(X_,X[i],mode='same') for i in range(shx[-1])]).reshape(sh[0],sh[1],shx[-1])
#    out_ = np.array([np.where(X > 0.,out[:,:,i,:],0.) for i in range(sh[2])])
    out_ = np.zeros_like(out)
    for i in range(sh[3]):
        for j in range(sh[2]):
            if j==i:
                out_[:,:,j,i] = np.where(X[:,:,i]>0.,out[:,:,j,i],0.)#out[:,:,j,i],0.)#convolve2d(W[:,:,j,i],X_[:,:,j],mode='same')
#            else:
#                out[:,:,j,i] = 0.
#    out_ = np.where(out_>0.,out_,0.)
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
