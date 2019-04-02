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

def stacked_layer(x,ksize=3,fsize=128,psize=(8,8),weights=None):
    x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1,weights=weights)(x)
    x_2 = MaxPool2D(pool_size=psize)(x_1)
#    x_3 = Conv2D(filters=fsize,kernel_size=1,padding='same',strides=1)(x_1)
    x_3 = ReLU()(x_2)
    x_4 = BatchNormalization()(x_3)
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
        self.s2 = Dropout(rate=0.5)(stacked_layer(self.s1,ksize=7,fsize=128,psize=2)) # 16,16,10,128
        self.s3 = stacked_layer(self.s2,ksize=5,fsize=256,psize=2) # 4,4,5,256
        self.fc1 = Dropout(rate=0.5)(stacked_layer(self.s3,ksize=3,fsize=512,psize=2)) # 1,1,1,2048
        out = Conv2D(filters=5,kernel_size=3,padding='same')(self.fc1)
        self.max_out = GlobalMaxPooling2D()(out)
        
        model = Model(inputs=inputs,outputs=self.max_out)
        model.compile(optimizer='adam',
              loss='logcosh',
              metrics=['accuracy'])
        return model

    def probe_FCN(self):
        inputs = Input(shape=self.cube_size)
        self.s1_ = stacked_layer(inputs,ksize=11,fsize=64,psize=4,weights=self.s1.get_weights())
        if layer2output == '1':
            model = Model(inputs=inputs,outputs=self.s1_)
            return model
        self.s2_ = stacked_layer(self.s1_,ksize=7,fsize=128,psize=2,weights=self.s2.get_weights())
        if layer2output == '2':
            model = Model(inputs=inputs,outputs=self.s2_)
            return model
        self.s3_ = stacked_layer(self.s2_,ksize=5,fsize=256,psize=2,weights=self.s3.get_weights())
        if layer2output == '3':
            model = Model(inputs=inputs,outputs=self.s3_)
            return model
        self.fc1_ = stacked_layer(self.s3_,ksize=3,fsize=512,psize=2,weights=self.fc1.get_weights())
        if layer2output == '4':
            model = Model(inputs=inputs,outputs=self.fc1_)
            return model
        out = Conv2D(filters=5,kernel_size=3,padding='same')(self.fc1_)
        self.max_out = GlobalMaxPooling2D()(out)
        if layer2output == '5':
            model = Model(inputs=inputs,outputs=self.max_out)
            return model

        
    def train(self,data_dict,epochs=10000,batch_size=12,scalar_=1e5,fgcube=None):
        loss_arr_t = []
        loss_arr_v = []
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
        data_dict_ = hf.scale_sample(data_dict)
        print('Normalizing scaled data cubes...')
        data_dict_ = hf.normalize(data_dict_)
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
            rnd_ind_t = np.random.choice(range(len(train_labels)),size=batch_size)
            rnd_ind_v = np.random.choice(range(len(val_labels)),size=batch_size)
            fcn_loss = self.fcn_model.train_on_batch(train_data[rnd_ind_t],train_labels[rnd_ind_t])
            val_loss = self.fcn_model.test_on_batch(val_data[rnd_ind_v],val_labels[rnd_ind_v])
            loss_arr_t.append(fcn_loss[0])
            loss_arr_v.append(val_loss[0])
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
