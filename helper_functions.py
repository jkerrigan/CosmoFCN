import numpy as np
from glob import glob
import re
import h5py
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('AGG')
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl
rc('text',usetex=True)
pl.rcParams.update({'font.size':18})
import socket
import os

def load_21cmCubes():
    # Cubes are in shape 512*512*30
    # Output will be in (X,Y,Z) = (512,512,30)
#    with h5py.File('/pylon5/as5fp5p/plaplant/21cm/t21_snapshots_downsample.hdf5') as f:
#    with h5py.File('/data4/plaplant/21cm/t21_snapshots_downsample.hdf5') as f:
    host = socket.getfqdn()
    print('Looking for data according to hostname: {}'.format(host))
    if host.rfind('intrepid') > 0:
        file_ = '/data4/plaplant/21cm/t21_snapshots_downsample_vary_both.hdf5'
    elif host.rfind('brown') > 0:
        file_ = os.path.expanduser('~/data/shared/LaPlanteSims/t21_snapshots_downsample_vary_both.hdf5')
#    file_ = './21cmFastSlices.hdf5'
    with h5py.File(file_) as f:
        print(f.keys())
        data_dict = {}
        data_dict['redshifts'] = f['Data']['layer_redshifts'][...]
        data_dict['data'] = np.asarray([cube.T for cube in f['Data']['t21_snapshots'][...]])
        data_dict['labels'] = f['Data']['snapshot_labels'][...][:,:3]
        data_dict['eor_amp'] = np.max(data_dict['data'][0])
    print('Dataset size {0}'.format(np.shape(data_dict['data'])))
    print('Label size {0}'.format(np.shape(data_dict['labels'])))
    return data_dict

def load_21cmCubes_2(file_ = None):
    if file_:
        with h5py.File(file_) as f:
            data_dict = {}
            data_dict['data'] = f['Data']['t21_snapshots'][...]
            data_dict['labels'] = f['Data']['snapshot_labels'][...][:,:3]
    else:
        print('No file given.')
    return data_dict

def load_FGCubes(file_):
    x = np.load(file_)
    return x['cubes']

def redshift_avg(z0,z_diff=0.4):
    freqs = np.linspace(20.,300.,1024)
    z = 1420./freqs - 1.
    adj_z_diff = np.abs([z[i+1]-z_i for i,z_i in enumerate(z) if i <len(z)-1])
    z0_ind = np.argmin(np.abs(z0-z))
    adj_z_diff_high = [i for i in range(100) if np.abs(np.sum(adj_z_diff[z0_ind:z0_ind+i])-z_diff/2.) < 0.1]
    adj_z_diff_low = [i for i in range(100) if np.abs(np.sum(adj_z_diff[z0_ind-i:z0_ind])-z_diff/2.) < 0.1]
    high_freq = freqs[z0_ind+adj_z_diff_high[0]]
    low_freq = freqs[z0_ind-adj_z_diff_low[0]]
    return low_freq,high_freq

#def z_block(f1,f2):
#    freqs = np.linspace(20.,300.,1024)
#    f1_ind = np.argmin(np.abs(f1-freqs))
#    f2_ind = np.argmin(np.abs(f2-freqs))
#    block = np.zeros(1024)
#    block[f1_ind:f2_ind] = 1.
#    return block

def fg_cube(n,mean_amp):
    # RA,DEC,Freq.
    cube = np.zeros((512,512,1024))
    freqs = np.linspace(0.02,0.3,1024)
    ras = np.random.choice(range(512),size=n)
    decs = np.random.choice(range(512),size=n)
    indices = np.random.choice(np.linspace(-0.5,0.),size=n)
    fluxes = np.random.choice(np.linspace(0.1*mean_amp,mean_amp),size=n)
    for ((i,f),(r,d)) in zip(zip(indices,fluxes),zip(ras,decs)):
        cube[r,d,:] += f*(freqs/0.15)**(i)
    return cube

def mean_z(cube,mask):
    return np.mean(np.multiply(cube,mask),axis=-1)
    #return np.mean(cube*mask,axis=-1)

def gaussian(nx,ny,std):
    canvas = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            canvas[int(i),int(j)] = np.exp(-((i-nx/2)**2 + (j-ny/2)**2)/(2.*std**2))
    return canvas

def convolve(sky,beam):
    SKY = np.fft.fft2(sky)
    BEAM = np.fft.fft2(beam)
    return np.fft.ifft2(SKY*BEAM).real

#def build_fg_z_cube(rs,eor_amp,scalar=1e5):
#    n_sources = int(np.random.normal(loc=8000,scale=2000))
#    cube = fg_cube(n_sources,scalar*eor_amp)
#    rs_freqs = list(map(redshift_avg,rs))
#    z_mask = list(map(z_block,rs_freqs))
#    fg_z_cube = list(map(mean_z,len(z_mask)*[cube],z_mask)) #[mean_z(cube,zm) for zm in z_mask]
#    gauss = gaussian(512,512,10.)
#    dirty_cube = np.array(list(map(convolve,fg_z_cube,len(rs)*[gauss]))).T
#    return dirty_cube

def random_perm(x):
    rnd = np.random.rand()
    if rnd < .3:
        return x[::-1,:,:]
    elif rnd > .3 and rnd < .6:
        return x[:,::-1,:]
    else:
        return x[::-1,::-1,:]

def scale_sample(data_dict,fgcube=None):
    # dataset comes in sizes of 512x512x30 (2Gpc,2Gpc,30 z slices)
    # we want to subsample this space to sizes < 1Gpc
    new_dict = {}
    scale = np.random.choice(range(64,256,30)) #32 minimum because of pooling operations (min 62.5 Gpc)
    print('Sampled to the scale of {} Mpc'.format(2000.*scale/512.))
    print('Length of data {}'.format(len(data_dict['data'])))
    print('Scale {}'.format(scale))
    dataset_size = np.shape(data_dict['data'])[0]
    if fgcube:
        if np.random.rand() > .0:
            rnd_fgs = np.random.choice(range(len(data_dict['foregrounds'])),size=len(data_dict['data']))
            fgs_realize = list(map(random_perm,np.array(data_dict['foregrounds'])[rnd_fgs]))
            print('FG Realizations {}'.format(len(fgs_realize)))
            combined_cubes = np.add(data_dict['data'],fgs_realize)
        else:
            combined_cubes = data_dict['data']
    else:
        combined_cubes = data_dict['data']
    scales = len(data_dict['data'])*[scale]
    print('.............')
    print('Combined cube size {}'.format(np.shape(combined_cubes)))
    data_arr = list(map(scale_,combined_cubes,scales))
    new_dict['data'] = data_arr
    new_dict['labels'] = data_dict['labels']
    new_dict['redshifts'] = data_dict['redshifts']
    if fgcube:
        new_dict['foregrounds'] = data_dict['foregrounds']
    return new_dict

def normalize(data_dict):
    def standard_(x):
        return (x - np.mean(x))/np.std(x)#(np.max(x)-np.min(x))
    try:
        data_dict['data'] = np.asarray(list(map(standard_,data_dict['data'])))
        #norm_dict['labels'] = data_dict['labels']
        return data_dict
    except:
        return standard_(data_dict)

def scale_(data,scale):
    s_x = np.random.choice(range(512-scale))
    s_y = np.random.choice(range(512-scale))
    data_ = data[s_x:scale+s_x,s_y:s_y+scale,:]
    rnd = np.random.rand()
#    if rnd > 0.3:
#        data_ = data_[::-1,:,:]
#    elif rnd > 0.3 and rnd <= 0.6:
#        data_ = data_[:,::-1,:]
#    else:
#        data_ = data_[::-1,::-1,:]
    #if np.random.rand() > .5:
    #    data_ += 0.05*np.std(data_)*np.random.randn(*np.shape(data_))
    return np.asarray(data_)

def pull_by_freq(x):
    return [x[:,:,i] for i in range(len(x[0,0,:]))]

def expand_cubes(dict_):
    # We want to exand the cubes and augment to increase 
    # the number of realizations we have
    data_ = []
    labels = []
    for key in dict_.keys():
        # Augment data
        for i in range(10):
            rnd = np.random.rand()
            if rnd < .2:
                data_aug = dict_[key][::-1,:,:]
            elif rnd > .2 and rnd <= .4:
                data_aug = dict_[key][:,::-1,:]
            elif rnd > .4 and rnd < .6:
                data_aug = dict_[key][::-1,::-1,:]
            elif rnd > .6 and rnd < .8:
                data_aug = dict_[key][::-1,:,::-1]
            else:
                data_aug = dict_[key][:,::-1,::-1]
                
            if np.random.rand() > 0.5:
                data_aug += 0.1*np.std(data_aug)*np.random.randn(*np.shape(data_aug))
            data_.append(data_aug)
            labels.append(key)

    # Grab random permutations of 2D slices
    data_ = np.expand_dims(np.array(list(map(pull_by_freq,data_))).reshape(-1,256,256),axis=-1)
    labels = np.array([128*[label] for label in labels]).reshape(-1)
    return data_,labels

def empirical_error_plots(t1_arr,p1_arr,err_arr,param,fname,spec=None):
    # t1_err should all be the same
    p1_max = np.max(p1_arr)
    p1_min = np.min(p1_arr)
    err_min = np.min(err_arr)
    
    pl.figure()
    ax1 = pl.subplot(211)
#    ax1.set_title(param)
    ax1.axvline(np.mean(t1_arr),0.,1.,linestyle='--',color='black',label='{0}$_{{pred}}$'.format(param))
    ax1.axvline(np.mean(p1_arr),0.,1.,linestyle='--',color='red',label='${0}$'.format(param))
    
#    ax1.text(np.mean(t1_arr)*0.95,0.1,r'${0}$'.format(param),rotation=90.)
#    ax1.text(np.mean(p1_arr)*0.95,0.1,r'{0}$_{{pred}}$'.format(param),rotation=90.)
    ax1.errorbar(p1_arr,np.linspace(0.,1,len(p1_arr)),xerr=err_arr,fmt='k.',markersize=2,ecolor='b',alpha=.4)
    ax1.set_xlim(p1_min*0.95,p1_max*1.05)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
#    pl.xlabel('Predicted {}'.format(param))
    
    pl.subplot(212)
    std_rnd = np.round(float(np.std(p1_arr)),2)
    mean_rnd = np.round(float(np.mean(err_arr)),2)
    pl.hist(p1_arr,histtype='step',fill=False,label='$\sigma_{e}$'+': {0:.2f}'.format(std_rnd))
    pl.hist(p1_arr,histtype='step',fill=False,label='$\sigma_{EKF}$'+': {0:.2f}'.format(mean_rnd))
#    pl.text(p1_min*1.05,20,r'$\sigma_{e}$'+': {0:.2f}'.format(std_rnd))
#    pl.text(p1_min*1.05,10,r'$\bar{\sigma}_{EKF}$'+': {0:.2f}'.format(mean_rnd))
    pl.xlim(p1_min*0.95,p1_max*1.05)
    pl.xlabel('Predicted {}'.format(param))
    pl.legend()
    pl.tight_layout()

#    pl.subplot(313)
#    pl.hist(err_arr)
#    pl.text(err_min*1.05,0.5,'avg: '+str(np.round(np.mean(err_arr),4)))
#    pl.axvline(np.round(np.std(p1_arr),4),linestyle='--',label='Empirical Std')
#    pl.xlabel('Uncertainties')
#    pl.legend()
    pl.savefig('empirical_errs_{}.pdf'.format(fname))
        
def plot_cosmo_params(t1_arr,p1_arr,err_arr,param,fname,spec=None):
    t1_arr = np.array(t1_arr)
    p1_arr = np.array(p1_arr)
    err_arr = np.array(err_arr)

    fig = pl.figure()
    gs1 = gridspec.GridSpec(3,1)
    ax1 = fig.add_subplot(gs1[:2])
    ax2 = fig.add_subplot(gs1[2])
#    ax3 = fig.add_subplot(gs1[2,1])
#    if spec is not None:
        #spec is an array with differenced midpoint,mean z
    s = spec/5.
#    else:
#        s = 5
    #print(spec)
#    ax1.errorbar(t1_arr,p1_arr,yerr=err_arr,alpha=0.5,fmt='.',color='black')
    ax1.scatter(t1_arr,p1_arr,c='black',s=s,alpha=0.5)
    ideal = np.linspace(0.8*np.min(t1_arr),1.2*np.max(t1_arr),10)
    ax1.set_ylim(np.min(p1_arr-err_arr)*0.9,np.max(p1_arr+err_arr)*1.1)
    ax1.plot(ideal,ideal,'r--')
    ax1.set_xlabel(r'')
    ax1.set_xticklabels([])
    ax1.locator_params(nbins=5)
#    pl.xticks(size=15)
#    pl.yticks(size=15)
    ax1.set_ylabel(r'Predicted {}'.format(param))

    ax2.plot(ideal,len(ideal)*[0.],'r--')
    err = 100.*(1-np.array(p1_arr)/np.array(t1_arr))

    X,Y = np.mgrid[0.80*np.min(t1_arr):1.2*np.max(t1_arr):100j, 0.80*np.min(p1_arr):1.2*np.max(p1_arr):100j]
    dense_grid = np.vstack([X.ravel(), Y.ravel()])
    data_arr = np.vstack([t1_arr,p1_arr])
    
#    kde = gaussian_kde(data_arr)
#    Z = np.reshape(kde(dense_grid).T,X.shape)
#    ax1.imshow(np.rot90(Z),cmap=pl.cm.gist_earth_r,extent=[0.95*np.min(t1_arr),1.05*np.max(t1_arr),0.95*np.min(p1_arr),1.05*np.max(p1_arr)],aspect='auto',alpha=0.8)
    ax1.set_xlim(0.95*np.min(t1_arr),1.05*np.max(t1_arr))
#    ax1.set_ylim(0.95*np.min(p1_arr),1.05*np.max(p1_arr))
    
    ax2.scatter(t1_arr,100.*(1-np.array(p1_arr)/np.array(t1_arr)),c='black',s=s,alpha=0.5)
    #ax2.plot(t1_arr,100.*(1-np.array(p1_arr)/np.array(t1_arr)),'k.',alpha=0.5)

    diffPredicted = 100.*(1-np.array(p1_arr)/np.array(t1_arr))
    ax2.set_ylim(np.min(diffPredicted)*0.95,np.max(diffPredicted)*1.05)
    ax2.locator_params(nbins=5)
    ax2.set_xlim(0.95*np.min(t1_arr),1.05*np.max(t1_arr))
    ax2.set_ylabel(r'\% error')
    ax2.set_xlabel(r'True {}'.format(param))
    #ax1.set_title(r'{}'.format(param))

#    density = kde(np.linspace(-30,30,100))
#    ax3.plot(density,np.linspace(-30,30,100))
    pl.tight_layout()
    pl.savefig('{}.pdf'.format(fname),dpi=300)
    pl.close()



if __name__ == "__main__":
    print('Helper function script.')
    load_21cmCubes()
