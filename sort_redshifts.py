import h5py
import numpy as np

'''
input z: (double) boundary redshift value z along which to split data
input mode: (int) that represents what cosmological parameter to split by (0 = ; 1= ; 2= ;) 
input filename: (str) name of saved split datafile
input datapath (str) name of hdf5 file to access data from
Saves a h5 file with data split between data with (mode) below (z) and (mode) above (z)
'''

def splitData(z=8., mode=0, basedir='/users/jsolt/data/shared/',datapath='/users/jsolt/data/shared/LaPlanteSims/t21_snapshots_downsample_vary_both.hdf5'):
	
	fnameh = basedir + 'sort_zcut{}_high.h5'.format(z)
	fnamel = basedir + 'sort_zcut{}_low.h5'.format(z)

	#find data at datapath and copy snapshots, labels into local vars
	print('Accessing data...')
	with h5py.File(datapath) as f:
		labels = f['Data']['snapshot_labels'][:,:]
		x = transposeCubes(f)	

	#make 2 new or clear 2 existing h5py files 
	high = h5py.File(fnameh,'w')
	low = h5py.File(fnamel,'w')
	try:
		high['Data']
	except:
		high.create_group('Data')
	try:
		low['Data']
	except:
		low.create_group('Data')
	high['Data'].clear()
	low['Data'].clear()

	highsnaps = []
	highlabels = []
	lowsnaps = []
	lowlabels = []
	
	print('Splitting data...')
	#split sample by z
	for i in range(0,len(x)):
		if labels[i,mode] > z:
			highsnaps.append(x[i])
			highlabels.append(labels[i,:])
		else:
			lowsnaps.append(x[i])
			lowlabels.append(labels[i,:])
		
	print('Saving data...')
	#save data in file
	high['Data']['t21_snapshots'] = highsnaps
	high['Data']['snapshot_labels']= highlabels
	low['Data']['t21_snapshots'] = lowsnaps
	low['Data']['snapshot_labels']= lowlabels

	high.close()
	low.close()

	print("Split " + str(len(highsnaps)) + " high and " + str(len(lowsnaps)) + " low.")


'''
Input f: the h5py file being considered
Returns an array of transposed cubes
'''
def transposeCubes(f):
    	return np.asarray([cube.T for cube in f['Data']['t21_snapshots'][...]])


if __name__ == '__main__':
	splitData(z=8.)
