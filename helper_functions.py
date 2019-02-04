import numpy as np
from glob import glob
import re
import h5py

def load_21cmCubes():
    # Cubes are in shape 512*512*20
    # Output will be in (X,Y,Z) = (512,512,20)
#    with h5py.File('/pylon5/as5fp5p/plaplant/21cm/t21_snapshots_downsample.hdf5') as f:  
    with h5py.File('/data4/plaplant/21cm/t21_snapshots_downsample_vary_both.hdf5') as f:
        print f.keys()
        data_dict = {}
        data_dict['redshifts'] = f['Data']['layer_redshifts'][...]
        data_dict['data'] = [cube.T for cube in f['Data']['t21_snapshots'][...]]
        data_dict['labels'] = f['Data']['snapshot_labels'][...]
        data_dict['eor_amp'] = np.max(data_dict['data'][0])
    print('Dataset size {0}'.format(np.shape(data_dict['data'])))
    print('Label size {0}'.format(np.shape(data_dict['labels'])))
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

def z_block((f1,f2)):
    freqs = np.linspace(20.,300.,1024)
    f1_ind = np.argmin(np.abs(f1-freqs))
    f2_ind = np.argmin(np.abs(f2-freqs))
    block = np.zeros(1024)
    block[f1_ind:f2_ind] = 1.
    return block

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

def build_fg_z_cube(rs,eor_amp,scalar=1e5):
    n_sources = int(np.random.normal(loc=8000,scale=2000))
    cube = fg_cube(n_sources,scalar*eor_amp)
    rs_freqs = list(map(redshift_avg,rs))
    z_mask = list(map(z_block,rs_freqs))
    fg_z_cube = list(map(mean_z,len(z_mask)*[cube],z_mask)) #[mean_z(cube,zm) for zm in z_mask]
    gauss = gaussian(512,512,10.)
    dirty_cube = np.array(list(map(convolve,fg_z_cube,len(rs)*[gauss]))).T
    return dirty_cube

def random_perm(x):
    rnd = np.random.rand()
    if rnd < .3:
        return x[::-1,:,:]
    elif rnd > .3 and rnd < .6:
        return x[:,::-1,:]
    else:
        return x[::-1,::-1,:]

def scale_sample(data_dict,fgcube=None):
    # dataset comes in sizes of 512x512x20 (2Gpc,2Gpc,30 z slices)
    # we want to subsample this space to sizes < 1Gpc
    new_dict = {}
    scale = np.random.choice(range(32,512,10)) #32 minimum because of pooling operations (min 62.5 Gpc)
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
        return (x-np.mean(x))/np.std(x)
    try:
        data_dict['data'] = list(map(standard_,data_dict['data']))
        return data_dict
    except:
        return standard_(data_dict)

def scale_(data,scale):
    s_x = np.random.choice(range(512-scale))
    s_y = np.random.choice(range(512-scale))
    data_ = data[s_x:scale+s_x,s_y:s_y+scale,:]
    rnd = np.random.rand()
    if rnd > 0.3:
        data_ = data_[::-1,:,:]
    elif rnd > 0.3 and rnd <= 0.6:
        data_ = data_[:,::-1,:]
    else:
        data_ = data_[::-1,::-1,:]
    if np.random.rand() > .5:
        data_ += 0.05*np.std(data_)*np.random.randn(*np.shape(data_))
    return data_

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


def plot_cosmo_params(t1_arr,p1_arr,param,fname):
    from matplotlib import rc
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as pl
    rc('text',usetex=True)
    fig = pl.figure()
    gs1 = gridspec.GridSpec(3,1)
    ax1 = fig.add_subplot(gs1[:2])
    ax2 = fig.add_subplot(gs1[2])
    ax1.scatter(t1_arr,p1_arr)
    ideal = np.linspace(np.min(t1_arr),np.max(t1_arr),10)
    ax1.plot(ideal,ideal,'--')
    ax1.set_xlabel(r'')
    ax1.set_ylabel(r'Predicted {0}'.format(param))

    ax2.plot(t1_arr,100.*(1-np.array(p1_arr)/np.array(t1_arr)),'.')
    ax2.set_ylim(-20.,20.)
    ax2.set_ylabel(r'\% error')
    ax2.set_xlabel(r'True {0}'.format(param))
    ax1.set_title(r'{0}'.format(param))
    pl.savefig('{0}.png'.format(fname))
    pl.close()




if __name__ == "__main__":
    print('Helper function script.')
    load_21cmFastCubes()
