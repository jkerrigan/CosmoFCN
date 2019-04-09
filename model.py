from keras.layers import ReLU,BatchNormalization
from keras.layers import Conv2D,MaxPool2D,GlobalMaxPooling2D
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

def stacked_layer(x,ksize=3,fsize=128,psize=(8,8),weights=None,trainable=True):
    if weights == None:
        x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,trainable=trainable)(x)
        x_2 = MaxPool2D(pool_size=psize)(x_1)
        x_3 = BatchNormalization()(x_2)
        x_4 = ReLU()(x_3)
    else:
        x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,weights=weights[:2],trainable=trainable)(x)
        x_2 = MaxPool2D(pool_size=psize)(x_1)
        x_3 = BatchNormalization(weights=weights[2:])(x_2)
        x_4 = ReLU()(x_3)
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
        self.s1 = stacked_layer(inputs,ksize=11,fsize=64,psize=4) # 64,64,10,64
        #s1_ms = Dropout(rate=0.5)(stacked_layer(inputs,ksize=7,fsize=64,psize=4))
        #s1_ls = Dropout(rate=0.5)(stacked_layer(inputs,ksize=3,fsize=64,psize=4))
        #xs1_ = concatenate([s1_ss,s1_ms],axis=-1)
        #s1 = concatenate([s1_,s1_ls],axis=-1)
        self.s2 = Dropout(rate=0.2)(stacked_layer(self.s1,ksize=7,fsize=128,psize=2)) # 16,16,10,128
        self.s3 = stacked_layer(self.s2,ksize=5,fsize=256,psize=2) # 4,4,5,256
        self.fc1 = Dropout(rate=0.2)(stacked_layer(self.s3,ksize=3,fsize=512,psize=2)) # 1,1,1,2048
        self.out = Conv2D(filters=3,kernel_size=3,padding='same')(self.fc1)
        self.max_out = GlobalMaxPooling2D()(self.out)
        
        model = Model(inputs=inputs,outputs=self.max_out)
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
        return model

    def probe_FCN(self,layer2output=None,weights=None):
        inputs = Input(shape=self.cube_size)
        print('w1',np.shape(weights))
        self.s1_ = stacked_layer(inputs,ksize=11,fsize=64,psize=4,weights=weights[:6])
        if layer2output == '1':
            model = Model(inputs=inputs,outputs=self.s1_)
            return model
#        print('w2',np.shape(weights[1]))
        self.s2_ = stacked_layer(self.s1_,ksize=7,fsize=128,psize=2,weights=weights[6:12])
        if layer2output == '2':
            model = Model(inputs=inputs,outputs=self.s2_)
            return model
#        print('w3',np.shape(weights[2]))
        self.s3_ = stacked_layer(self.s2_,ksize=5,fsize=256,psize=2,weights=weights[12:18])
        if layer2output == '3':
            model = Model(inputs=inputs,outputs=self.s3_)
            return model
        self.fc1_ = stacked_layer(self.s3_,ksize=3,fsize=512,psize=2,weights=weights[18:24])
        if layer2output == '4':
            model = Model(inputs=inputs,outputs=self.fc1_)
            return model
        self.out_ = Conv2D(filters=3,kernel_size=3,padding='same',weights=weights[24:26])(self.fc1_)
        self.max_out = GlobalMaxPooling2D(weights=weights[26:30])(self.out_)
        if layer2output == '5':
            model = Model(inputs=inputs,outputs=self.max_out)
            return model
#0,1,6,7,12,13,18,19,24,25
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
        loss_arr_v = []
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
        data_dict_ = hf.normalize(data_dict) # normalize all data once and first
        data = data_dict_['data']
        labels = data_dict_['labels']
        redshifts = data_dict_['redshifts']
        length = len(labels)
        train_data = np.array(data[:int(length*0.8)])
        train_labels = np.array(labels[:int(length*0.8)])
        
        val_data = np.array(data[int(length*0.8):])
        val_labels = np.array(labels[int(length*0.8):])
        
        #fcn_model.fit(self.data,self.labels)
        gc.enable() #attempt garbage collection to release resources
        for e in range(epochs):
            print(e)
            rnd_ind_t = np.random.choice(range(len(train_labels)),size=batch_size)
            rnd_ind_v = np.random.choice(range(len(val_labels)),size=batch_size)

            train_scale = train_data[rnd_ind_t]
            val_scale = val_data[rnd_ind_v]
#            train_dict = {'data':train_scale,'labels':train_labels[rnd_ind_t],'redshifts':[]}
#            val_dict = {'data':val_scale,'labels':val_labels[rnd_ind_v],'redshifts':[]}
#            train_dict = hf.scale_sample(train_dict)
#            val_dict = hf.scale_sample(val_dict)
#            print('Train data shape: ',np.shape(train_dict['data']))

            fcn_loss = self.fcn_model.train_on_batch(np.array(train_data[rnd_ind_t]),train_labels[rnd_ind_t])
            val_loss = self.fcn_model.test_on_batch(np.array(val_data[rnd_ind_v]),val_labels[rnd_ind_v])
            loss_arr_t.append(fcn_loss[0])
            loss_arr_v.append(val_loss[0])
#            del(val_dict)
#            del(train_dict)
            print('Epoch: {0} Train Loss: {1} Validation Loss: {2}'.format(e,fcn_loss[0],val_loss[0]))
            if e % 500==0 and e!=0:
                del(train_data)
                del(val_data)
                print('Rescaling down new cubes...')
                data_dict_ = hf.scale_sample(data_dict)
                print('Normalizing new scaled data cubes...')
                data_dict_ = hf.normalize(data_dict_)
                data = np.copy(data_dict_['data'])
                del(data_dict_)
                train_data = np.array(data[:int(length*0.8)])
                val_data = np.array(data[int(length*0.8):])
        plot_loss(self.model_name,range(epochs),loss_arr_t,loss_arr_v)
        return self.fcn_model

    def save(self):
        print('Saving trained model...')
        self.fcn_model.save_weights(self.model_name+'.h5')
        model_json = self.fcn_model.to_json()
        with open(self.model_name+'.json', "w") as json_file:
            json_file.write(model_json)
        print('Model saved.')
        
    def load(self):
        json_file = open(self.model_name+'.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.fcn_model = model_from_json(loaded_model_json)
        self.fcn_model.load_weights(self.model_name+'.h5')
        print('Model loaded.')

def plot_loss(model_name,iters,train_loss,val_loss):
    pl.plot(iters,train_loss,label='Training loss')
    pl.plot(iters,val_loss,label='Evaluation loss')
    pl.xlabel('Iterations')
    pl.ylabel('loss')
    pl.legend()
    pl.savefig(model_name+'_loss.pdf',dpi=300)
