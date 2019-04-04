from model import FCN21CM
from helper_functions import load_21cmCubes,plot_cosmo_params,build_fg_z_cube
import helper_functions as hf
import matplotlib
matplotlib.use('AGG')
import numpy as np
import pylab as pl
import sys
from EKF import EKFCNN

save = False
EKF = True
# Load data

data_dict = load_21cmCubes()
#data = data_dict['data']
#labels = data_dict['labels']

fcn = FCN21CM(lr=0.003,model_name='ProofOfConcept_Standardize')
fcn.load()
#fcn.train(data_dict,epochs=1000,batch_size=32,scalar_=1e0,fgcube=None)
#fcn.save()


# zmid, delta_z, zmean, alpha, kb
p1_arr = []
p2_arr = []
p3_arr = []
p4_arr = []
p5_arr = []

t1_arr = []
t2_arr = []
t3_arr = []
t4_arr = []
t5_arr = []

ssize = []

p1_arr_err = []
p2_arr_err = []
p3_arr_err = []
p4_arr_err = []
p5_arr_err = []

if EKF:
    probes,weights = fcn.get_probes()
    ekf_model = EKFCNN(probes,weights)
    ekf_model.run_EKF(data_dict['data'][:300])
    

for i in range(200):
    print('Predicting on sample {0}')
    redshifts = data_dict['redshifts']
    eor_amp = data_dict['eor_amp']
    #if False:#np.random.rand()>1.1:
    #    fgs = build_fg_z_cube(redshifts,eor_amp,scalar)
    #    combined_cubes = np.add(data_dict['data'][-i],fgs)
    #else:
    combined_cubes = data_dict['data'][-np.mod(i,200)]
    print(np.shape(combined_cubes))
    rnd_scale = np.random.choice(range(64,512,1))
    data_sample = hf.scale_(hf.normalize(combined_cubes),rnd_scale).reshape(1,rnd_scale,rnd_scale,30)
    label_sample = data_dict['labels'][-np.mod(i,200)]
    print('scaled sample shape',np.shape(data_sample))
    predict = fcn.fcn_model.predict(data_sample)[0]

    predict_err = ekf_model.pred_uncertainty(data_sample)
    
    p1_arr.append(predict[0])
    p2_arr.append(predict[1])
    p3_arr.append(predict[2])
    p4_arr.append(predict[3])
    p5_arr.append(predict[4])
    ssize.append(rnd_scale)

    t1_arr.append(label_sample[0])
    t2_arr.append(label_sample[1])
    t3_arr.append(label_sample[2])
    t4_arr.append(label_sample[3])
    t5_arr.append(label_sample[4])
    print('NamesL {}'.format(['midpoint','duration','meanz','alpha','k0']))
    print('Predicted Error: {}'.format(predict_err))
    p1_arr_err.append(predict_err[0])
    p2_arr_err.append(predict_err[1])
    p3_arr_err.append(predict_err[2])
    p4_arr_err.append(predict_err[3])
    p5_arr_err.append(predict_err[4])
    
pl.plot(np.array(ssize)*2000./512.,np.abs(np.array(t2_arr)-np.array(p2_arr)),'.')
pl.xlabel('Cube Size (Mpc)')
pl.ylabel('error %')
pl.savefig('ErrorVsSize.pdf',dpi=300)
    
predict_arr = [p1_arr,p2_arr,p3_arr,p4_arr,p5_arr]
true_arr = [t1_arr,t2_arr,t3_arr,t4_arr,t5_arr]
error_arr = [p1_arr_err,p2_arr_err,p3_arr_err,p4_arr_err,p5_arr_err]
pnames = ['$z_{50\%}$','$\Delta z$','$\overline{z}$','alpha','$k_{0}$']
fnames = ['midpoint','duration','meanz','alpha','k0']

if save:
    np.savez('fg_scalar_{0}.npz'.format(scalar),true=true_arr,predicted=predict_arr,names=fnames)

for i,(p_,f_) in enumerate(zip(pnames,fnames)):
#    if f_ == 'duration':
    spec = 30.*(np.array(ssize)/256.)#np.exp(np.array(true_arr[0]) - np.array(true_arr[2]))
#    else:
#        spec = None
    plot_cosmo_params(true_arr[i],predict_arr[i],error_arr[i],p_,f_,spec=spec)
