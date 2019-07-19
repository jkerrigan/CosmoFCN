from keras.layers import ReLU,BatchNormalization
from keras.layers import Conv2D,MaxPool2D,GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Dense, Reshape, Multiply, Lambda, concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
import helper_functions as hf
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as pl
import gc

def stacked_layer(x,ksize=3,fsize=128,psize=(8,8),weights=None,trainable=True,batchnorm=True):
    if weights == None:
        if batchnorm:
            x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,trainable=trainable)(x)
            x_2 = MaxPool2D(pool_size=psize)(x_1)
            x_3 = BatchNormalization()(x_2)
            x_4 = ReLU()(x_3)
        else:
            x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,trainable=trainable)(x)
            x_2 = MaxPool2D(pool_size=psize)(x_1)
            x_4 = ReLU()(x_2)
    else:
        if batchnorm:
            x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,weights=weights[:2],trainable=trainable)(x)
            x_2 = MaxPool2D(pool_size=psize)(x_1)
            x_3 = BatchNormalization(weights=weights[2:])(x_2)
            x_4 = ReLU()(x_3)
        else:
            x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,weights=weights[:2],trainable=trainable)(x)
            x_2 = MaxPool2D(pool_size=psize)(x_1)
            x_4 = ReLU()(x_2)
    return x_4

class FCN21CM():
    def __init__(self,lr=0.003,model_name='Null'):
        optimizer = Adam(lr=lr)
        self.model_name = model_name
        self.X_size = None
        self.Y_size = None
        self.Z_size = 30
        self.cube_size = (self.X_size,self.Y_size,self.Z_size)
    
    def FCN(self):
        inputs = Input(shape=self.cube_size)

        self.s1 = stacked_layer(inputs,ksize=3,fsize=32,psize=4,batchnorm=False) # 64,64,10,64
        self.s2 = Dropout(rate=0.0)(stacked_layer(self.s1,ksize=3,fsize=64,psize=2,batchnorm=False)) # 16,16,10,128
        self.s3 = Dropout(rate=0.0)(stacked_layer(self.s2,ksize=3,fsize=128,psize=2,batchnorm=False)) # 4,4,5,256
        self.fc1 = Dropout(rate=0.0)(stacked_layer(self.s3,ksize=3,fsize=256,psize=2,batchnorm=False)) # 1,1,1,2048
        self.out = Dropout(rate=0.0)(Conv2D(filters=3,kernel_size=3,padding='same')(self.fc1))
        self.max_out = GlobalMaxPooling2D()(self.out)
        
        model = Model(inputs=inputs,outputs=self.max_out)
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
        return model

    def probe_FCN(self,layer2output=None,weights=None):
        inputs = Input(shape=self.cube_size)
        print('w1',np.shape(weights))
        if layer2output == '0':
            model = Model(inputs=inputs,outputs=inputs)
        self.s1_ = stacked_layer(inputs,ksize=3,fsize=32,psize=4,weights=weights[:2],batchnorm=False)
        if layer2output == '1':
            model = Model(inputs=inputs,outputs=self.s1_)
            return model
#        print('w2',np.shape(weights[1]))
        self.s2_ = stacked_layer(self.s1_,ksize=3,fsize=64,psize=2,weights=weights[2:4],batchnorm=False)
        if layer2output == '2':
            model = Model(inputs=inputs,outputs=self.s2_)
            return model
#        print('w3',np.shape(weights[2]))
        self.s3_ = stacked_layer(self.s2_,ksize=3,fsize=128,psize=2,weights=weights[4:6],batchnorm=False)
        if layer2output == '3':
            model = Model(inputs=inputs,outputs=self.s3_)
            return model
        self.fc1_ = stacked_layer(self.s3_,ksize=3,fsize=256,psize=2,weights=weights[6:8],batchnorm=False)
        if layer2output == '4':
            model = Model(inputs=inputs,outputs=self.fc1_)
            return model
        self.out_ = Conv2D(filters=3,kernel_size=3,padding='same',weights=weights[8:])(self.fc1_)
        self.max_out = GlobalMaxPooling2D()(self.out_)
        if layer2output == '5':
            model = Model(inputs=inputs,outputs=self.max_out)
            return model

    def get_probes(self):
        layers = [str(i) for i in range(1,6)]
        weight_layers = self.get_weights()
        pred_layers = [self.probe_FCN(layer,weights=weight_layers) for layer in layers]
        return pred_layers, weight_layers

    def get_weights(self):
        #print([weight.name for layer in self.fcn_model.layers for weight in layer.weights])
        #print([weight for layer in self.fcn_model.layers for weight in layer.get_weights()])
#        for w in self.fcn_model.get_weights():
#            print(np.shape(w))
        #weights_ = np.reshape(self.fcn_model.get_weights()[:16],(-1,4)) #[np.array((self.fcn_model.get_weights()[i],self.fcn_model.get_weights()[i+1])) for i,w in enumerate(self.fcn_model.get_weights()) if i in range(0,40,6)]
        #weights = [weights_,self.fcn_model.get_weights()[16:18],self.fcn_model.get_weights()[18:]] #5x4 and 6
        weights = [weight for layer in self.fcn_model.layers for weight in layer.get_weights()]
        print(np.shape(weights))
        print('Got weights')
#        weights = [model_layer.get_weights()[0] for model_layer in self.fcn_model]
        return weights
    
    def train(self,data_dict,epochs=10000,batch_size=12,scalar_=1e5,fgcube=None):
        loss_arr_t = []
        loss_arr_v = [] #want
        resizing=True
        if 'self.fcn_model' in globals():
            print('Model valid.')
        else:
            print('No model found, starting from scratch.')
            self.fcn_model = self.FCN()
        print(self.fcn_model.summary())
        print('Doing a 80/20 Dataset Split.')
#        print('Building several realizations of point source foregrounds...')
        if fgcube:
            fgs = hf.load_FGCubes(fgcube)
            data_dict['foregrounds'] = fgs
            del(fgs)
        elif fgcube == 'Generate':
            fgs = [hf.build_fg_z_cube(data_dict['redshifts'],eor_amp=data_dict['eor_amp'],scalar=scalar_) for i in range(50)]
            data_dict['foregrounds'] = fgs
            del(fgs)
        else:
            print('No foregrounds included.')
        print('Scaling down cubes...')
#        data_dict_ = hf.scale_sample(data_dict)
        print('Normalizing scaled data cubes...')
        data_dict = hf.normalize(data_dict) # normalize all data once and first
        data_dict_ = hf.scale_sample(data_dict)
        data = np.copy(data_dict_['data'])
        labels = np.copy(data_dict_['labels'])
        redshifts = np.copy(data_dict_['redshifts'])
        length = len(labels)
        train_data = np.array(data[:int(length*0.8)])
        train_labels = np.array(labels[:int(length*0.8)])
        
        val_data = np.array(data[int(length*0.8):])
        val_labels = np.array(labels[int(length*0.8):])
        epoch_inds_t = np.array(range(len(train_labels))).reshape(-1,batch_size)
        #fcn_model.fit(self.data,self.labels)
        gc.enable() #attempt garbage collection to release resources
        epoch_loss_t = []
        epoch_loss_v = []
        epoch_loss_vten = []
        for e in range(epochs):
            print('Training Completed : {0}%'.format(100.*e/(1.*epochs)))
            #print(e)
            #rnd_ind_t = np.random.choice(range(len(train_labels)),size=batch_size)
            epoch_inds_t = np.random.permutation(epoch_inds_t)
            for i in range(int(len(train_labels)/batch_size)):
                rnd_ind_v = np.random.choice(range(len(val_labels)),size=batch_size)

                #train_scale = train_data[rnd_ind_t]
                #val_scale = val_data[rnd_ind_v]
                #            train_dict = {'data':train_scale,'labels':train_labels[rnd_ind_t],'redshifts':[]}
                #            val_dict = {'data':val_scale,'labels':val_labels[rnd_ind_v],'redshifts':[]}
                #            train_dict = hf.scale_sample(train_dict)
                #            val_dict = hf.scale_sample(val_dict)
                #            print('Train data shape: ',np.shape(train_dict['data']))
                fcn_loss = self.fcn_model.train_on_batch(np.array(train_data[epoch_inds_t[i,:]]),train_labels[epoch_inds_t[i,:]])
                val_loss = self.fcn_model.test_on_batch(np.array(val_data[rnd_ind_v]),val_labels[rnd_ind_v])
                loss_arr_t.append(fcn_loss[0])
                loss_arr_v.append(val_loss[0])
                #            del(val_dict)
                #            del(train_dict)
            print('Epoch: {0} Train Loss: {1} Validation Loss: {2}'.format(e,np.mean(loss_arr_t),np.mean(loss_arr_v)))
            epoch_loss_t.append(np.mean(loss_arr_t))
            epoch_loss_v.append(np.mean(loss_arr_v))

            if e % 100==0:
                self.save(n=('_'+str(e)))
                epoch_loss_vten.append(np.mean(loss_arr_v))

            if resizing:
                del(train_data)
                del(val_data)
                #print('Rescaling down new cubes...')
                #data_dict_ = hf.normalize(data_dict)
                data_dict_ = hf.scale_sample(data_dict)
                #print('Normalizing new scaled data cubes...')
                data = np.copy(data_dict_['data'])
                del(data_dict_)
                train_data = np.array(data[:int(length*0.8)])
                val_data = np.array(data[int(length*0.8):])
        
        plot_loss(self.model_name,range(epochs),epoch_loss_t,epoch_loss_v)
        
        with open('target_model_epochs.txt','w') as f:
            f.write('Minimum loss: ' + str(min(epoch_loss_vten)) +'\n')
            f.write('Maximum loss: ' + str(max(epoch_loss_vten)) +'\n')

            lossval = self.findloss(losspercent=0.90,losslist=epoch_loss_vten)
            f.write('90% loss value: ' +str(epoch_loss_vten[lossval])+'\n')
            f.write('90% epoch: ' + str(lossval*5)+'\n')

            lossval = self.findloss(losspercent=0.85,losslist=epoch_loss_vten)
            f.write('85% loss value: ' +str(epoch_loss_vten[lossval])+'\n')
            f.write('85% epoch: ' + str(lossval*5)+'\n')

            lossval = self.findloss(losspercent=0.80,losslist=epoch_loss_vten)
            f.write('80% loss value: ' +str(epoch_loss_vten[lossval])+'\n')
            f.write('80% epoch: ' + str(lossval*5)+'\n')

            lossval = self.findloss(losspercent=0.75,losslist=epoch_loss_vten)
            f.write('75% loss value: ' +str(epoch_loss_vten[lossval])+'\n')
            f.write('75% epoch: ' + str(lossval*5)+'\n')

            lossval = self.findloss(losspercent=0.50,losslist=epoch_loss_vten)
            f.write('50% loss value: ' +str(epoch_loss_vten[lossval])+'\n')
            f.write('50% epoch: ' + str(lossval*5)+'\n')

            lossval = self.findloss(losspercent=0.25,losslist=epoch_loss_vten)
            f.write('25% loss value: ' +str(epoch_loss_vten[lossval])+'\n')
            f.write('25% epoch: ' + str(lossval*5)+'\n')

        return self.fcn_model



    def findloss(self, losspercent=0, losslist=None):
	
        invert = [1.0/x for x in losslist]
        invert[:] = [x - np.min(invert) for x in invert]
        invert[:] = [x / np.max(invert) for x in invert] 
        target = np.argmin([abs(losspercent-x) for x in invert]) 
        print('Loss percent: ' + str(losspercent)+'\n')
        print('Target: ' + str(losslist[target])+'\n')
        print(list(invert))
        return target


    def save(self,n=''):
        print('Saving trained model...')
        self.fcn_model.save_weights(self.model_name + n +'.h5')
        model_json = self.fcn_model.to_json()
        with open(self.model_name + n + '.json', "w") as json_file:
            json_file.write(model_json)
        print('Model saved.')
        
    def load(self):
        json_file = open(self.model_name+'.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.fcn_model = model_from_json(loaded_model_json)
        self.fcn_model.load_weights(self.model_name+'.h5')
        print('Model loaded.')

    def writesummary(self, filename='modelsummary.txt'):
        with open(filename,'a') as fh:
            self.fcn_model.summary(print_fn=lambda x: fh.write(x + '\n'))

def plot_loss(model_name,iters,train_loss,val_loss):
    pl.plot(iters,np.log10(train_loss),label='Training loss')
    pl.plot(iters,np.log10(val_loss),label='Evaluation loss')
    pl.xlabel('Number of Iterations')
    pl.ylabel('Log MSE Loss')
    pl.legend()
    pl.savefig(model_name+'_loss.pdf',dpi=300)
