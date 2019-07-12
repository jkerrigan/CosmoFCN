from model import FCN21CM
from helper_functions import load_21cmCubes,plot_cosmo_params
import helper_functions as hf
import matplotlib
matplotlib.use('AGG')
import numpy as np
import pylab as pl
from time import time
import sys
from EKF import EKFCNN
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
import datetime

save = False
EKF = True
# Load data

modelname = 'train_2500epochs_nobatchnorm'

#training='~/data/shared/v2_filtered_single_sample8.5.h5' 
training = '~/data/shared/LaPlanteSims/v2/t21_snapshots_filtered.hdf5'
predicting = '~/data/shared/LaPlanteSims/t21_snapshots_downsample_vary_both.hdf5'
#predicting = training

savedpath = os.getcwd()
if not os.path.isdir(modelname):
	os.mkdir(modelname)
os.chdir(modelname)

#data_dict = hf.load_21cmCubes(training)
#data = data_dict['data']
#labels = data_dict['labels']

fcn = FCN21CM(lr=0.003,model_name=modelname)
try:
    fcn.load()
except:
    print('Model load error.')
#fcn.train(data_dict,epochs=10000,batch_size=160,scalar_=1e0,fgcube=None)
#fcn.save()

'''
with open('modelsummary.txt','w') as f:
    f.write(str(datetime.datetime.today())+'\n\n')
    f.write('Model file name: ' + modelname + '\n')
    f.write('Training data: ' + training + '\n')
    f.write('Notes: Batch 160. training for 2500 epochs. no batch normalization\n')
    f.write('Model Summary: \n')

fcn.writesummary()
'''

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

#print('Running test...')
#t0 = time()
#hf.tf_scale(data_dict)
#print('Tensorflow implementation time: ',time() - t0)
'''
t0 = time()
hf.scale_sample(data_dict)
print('Numpy implementation time: ',time() - t0)
'''
data_dict_predict = hf.load_21cmCubes(predicting,partial=True)

if EKF:
    cov_num = 200
    rnd_scale = 256#np.random.choice(range(64,256,1))
    noise =  0.*np.random.normal(loc=0.,scale=2.*np.std(data_dict_predict['data'][0]),size=(cov_num,512,512,30))
#    dset_EKF = data_dict_predict['data'][:cov_num]# + noise
    dset_EKF = [data_dict_predict['data'][30] for i in range(cov_num)]
    print('EKF dataset size: {}'.format(np.shape(dset_EKF)))
    scaled_EKF_data = np.asarray(list(map(hf.scale_,list(map(hf.normalize,dset_EKF)),cov_num*[rnd_scale]))).reshape(cov_num,rnd_scale,rnd_scale,30)
    print('Scaled EKF data',np.shape(scaled_EKF_data))
    probes,weights = fcn.get_probes()
    ekf_model = EKFCNN(probes,weights)
    ekf_model.run_EKF(scaled_EKF_data)

newpath = modelname + '_data_2'
if not os.path.isdir(newpath):
	os.mkdir(newpath)
os.chdir(newpath)




snr = np.linspace(0.,0.,len(data_dict_predict['data']))
for i in range(100):#len(data_dict_predict['data'])):
    print('Predicting on sample {0}')
    redshifts = data_dict_predict['redshifts']
#    eor_amp = data_dict_predict['eor_amp']
    #if False:#np.random.rand()>1.1:
    #    fgs = build_fg_z_cube(redshifts,eor_amp,scalar)
    #    combined_cubes = np.add(data_dict['data'][-i],fgs)
    #else:
    combined_cubes = data_dict_predict['data'][30]#-np.mod(i,50)]
    print(np.shape(combined_cubes))
    rnd_scale = 256 #np.random.choice(range(64,256,1))
    #noise = np.zeros((512,512,30))#

    noise =  snr[i]*np.random.normal(loc=0.,scale=snr[i]*np.std(combined_cubes),size=(512,512,30))#snr[i]*np.std(combined_cubes)*np.random.rand(512,512,30)
    print('Data std: {}'.format(np.std(combined_cubes)))
    print('Noise std: {}'.format(np.std(noise)))
    #data_sample = np.expand_dims(combined_cubes,axis=0)
    data_sample = hf.scale_(hf.normalize(combined_cubes + noise),rnd_scale).reshape(1,rnd_scale,rnd_scale,30)
    label_sample = data_dict_predict['labels'][30]#-np.mod(i,200)]
    print(label_sample.shape)
    print('scaled sample shape',np.shape(data_sample))
    predict = fcn.fcn_model.predict(data_sample)[0]

    predict_err = ekf_model.pred_uncertainty(data_sample)
    print('Predicted Midpoint {0} Duration {1} Mean Z {2}'.format(*predict))
    p1_arr.append(predict[0])
    p2_arr.append(predict[1])
    p3_arr.append(predict[2])
#    p4_arr.append(predict[3])
#    p5_arr.append(predict[4])
    ssize.append(rnd_scale)

    t1_arr.append(label_sample[0])
    t2_arr.append(label_sample[1])
    t3_arr.append(label_sample[2])
#    t4_arr.append(label_sample[3])
#    t5_arr.append(label_sample[4])
    print('Names: {}'.format(['midpoint','duration','meanz','alpha','k0']))
    print('Predicted Error: {}'.format(predict_err))

    p1_arr_err.append(predict_err[0])
    p2_arr_err.append(predict_err[1])
    p3_arr_err.append(predict_err[2])
#    p4_arr_err.append(predict_err[3])
#    p5_arr_err.append(predict_err[4])
'''
pl.figure()
pl.plot(snr,p1_arr_err,label='Midpoint')
pl.plot(snr,p2_arr_err,label='Duration')
pl.plot(snr,p3_arr_err,label='Mean Z')
pl.legend()
pl.savefig('SNRvsUncertainty.pdf',dpi=300)
'''
pl.figure()
pl.plot(np.array(ssize)*2000./512.,np.abs(np.array(t2_arr)-np.array(p2_arr)),'.')
pl.xlabel('Cube Size (Mpc)')
pl.ylabel('error %')
pl.savefig('ErrorVsSize.pdf',dpi=300)
    
predict_arr = [p1_arr,p2_arr,p3_arr]#,p4_arr,p5_arr]
true_arr = [t1_arr,t2_arr,t3_arr]#,t4_arr,t5_arr]
error_arr = [p1_arr_err,p2_arr_err,p3_arr_err]#,p4_arr_err,p5_arr_err]
pnames = ['$z_{50\%}$','$\Delta z$','$\overline{z}$']#,'alpha','$k_{0}$']
fnames = ['midpoint','duration','meanz']#,'alpha','k0']

if save:
    np.savez('fg_scalar_{0}.npz'.format(scalar),true=true_arr,predicted=predict_arr,names=fnames)

np.savez('predictions_output.npz',targets=true_arr,predictions=predict_arr,names=fnames)

#error_arr = np.zeros_like(predict_arr)

for i,(p_,f_) in enumerate(zip(pnames,fnames)):
#    if f_ == 'duration':
    spec = 30.*(np.array(ssize)/256.)#np.exp(np.array(true_arr[0]) - np.array(true_arr[2]))
#    else:
#        spec = None
    plot_cosmo_params(true_arr[i],predict_arr[i],error_arr[i],p_,f_,spec=spec)
    hf.empirical_error_plots(true_arr[i],predict_arr[i],error_arr[i],p_,f_,spec=spec)
    hf.distribution_measure(true_arr[i],predict_arr[i],error_arr[i])

with open('predictsummary.txt','w') as f:
    f.write('Predicting on: ' + predicting + '\n\n')

os.chdir(savedpath)
