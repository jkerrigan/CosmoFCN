from model import FCN21CM
from helper_functions import load_21cmCubes,plot_cosmo_params,build_fg_z_cube
import helper_functions as hf
import matplotlib
matplotlib.use('AGG')
import numpy as np
import pylab as pl
import sys

save = False
# Load data

data_dict = load_21cmCubes()
#data = data_dict['data']
#labels = data_dict['labels']

fcn = FCN21CM()
fcn.load('NoFGModel.json')
fcn.train(data_dict)
fcn.save('NoFGModel.json')


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

for i in range(300):
    print('Predicting on sample {0}')
    redshifts = data_dict['redshifts']
    eor_amp = data_dict['eor_amp']
    if np.random.rand()>1.1:
        fgs = build_fg_z_cube(redshifts,eor_amp,scalar)
        combined_cubes = np.add(data_dict['data'][-i],fgs)
    else:
        combined_cubes = data_dict['data'][-i]
    print(np.shape(combined_cubes))
    data_sample = hf.normalize(np.expand_dims(combined_cubes,axis=0))
    label_sample = data_dict['labels'][-i]
    predict = fcn.fcn_model.predict(data_sample)[0]
    p1_arr.append(predict[0])
    p2_arr.append(predict[1])
    p3_arr.append(predict[2])
    p4_arr.append(predict[3])
    p5_arr.append(predict[4])

    t1_arr.append(label_sample[0])
    t2_arr.append(label_sample[1])
    t3_arr.append(label_sample[2])
    t4_arr.append(label_sample[3])
    t5_arr.append(label_sample[4])


predict_arr = [p1_arr,p2_arr,p3_arr,p4_arr,p5_arr]
true_arr = [t1_arr,t2_arr,t3_arr,t4_arr,t5_arr]

pnames = ['$z_{0}$','$\Delta z$','$\mu_{z}$','$\tau$','kb']
fnames = ['midpoint','duration','meanz','optical_depth','kb']

if save:
    np.savez('fg_scalar_{0}.npz'.format(scalar),true=true_arr,predicted=predict_arr,names=fnames)

for i,(p_,f_) in enumerate(zip(pnames,fnames)):
    plot_cosmo_params(true_arr[i],predict_arr[i],p_,f_)
