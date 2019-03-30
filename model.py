from keras.layers import LeakyReLU,BatchNormalization
from keras.layers import Conv2D,MaxPool2D,GlobalMaxPooling2D
from keras.layers import Input, Dense, Reshape, Multiply, Lambda, concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
import helper_functions as hf
import numpy as np
import gc

def stacked_layer(x,ksize=3,fsize=128,psize=(8,8)):
    x_1 = Conv2D(filters=fsize,kernel_size=ksize,padding='same',strides=1)(x)
    x_2 = MaxPool2D(pool_size=psize)(x_1)
#    x_3 = Conv2D(filters=fsize,kernel_size=1,padding='same',strides=1)(x_1)
    x_3 = LeakyReLU()(x_2)
    x_4 = BatchNormalization()(x_3)
    return x_4

class FCN21CM():
    def __init__(self,lr=0.003):
        optimizer = Adam(lr=lr)
        self.X_size = None
        self.Y_size = None
        self.Z_size = 30
        self.cube_size = (self.X_size,self.Y_size,self.Z_size)
    
    def FCN(self):
        inputs = Input(shape=self.cube_size)
        s1_ss = Dropout(rate=0.3)(stacked_layer(inputs,ksize=7,fsize=64,psize=2)) # 64,64,10,64
        s1_ms = Dropout(rate=0.3)(stacked_layer(inputs,ksize=5,fsize=64,psize=2))
        s1_ls = Dropout(rate=0.3)(stacked_layer(inputs,ksize=3,fsize=64,psize=2))
        s1_ = concatenate([s1_ss,s1_ms],axis=-1)
        s1 = concatenate([s1_,s1_ls],axis=-1)
        print(np.shape(s1))
        s2 = stacked_layer(s1,ksize=3,fsize=64,psize=2) # 16,16,10,128
        s3 = stacked_layer(s2,ksize=3,fsize=128,psize=2) # 4,4,5,256
        fc1 = stacked_layer(s3,ksize=3,fsize=256,psize=2) # 1,1,1,2048
        out = Conv2D(filters=5,kernel_size=3,padding='same')(fc1)
        max_out = GlobalMaxPooling2D()(out)
#        out = Reshape((2,))(max_out)
#        out = Lambda(lambda x: x)(out)#Multiply()([20,out])
        
        model = Model(inputs=inputs,outputs=max_out)
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
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

        return self.fcn_model

    def save(self,model='model.json'):
        print('Saving trained model...')
        self.fcn_model.save_weights(model.split('.')[0]+'.h5')
        model_json = self.fcn_model.to_json()
        with open(model, "w") as json_file:
            json_file.write(model_json)
        print('Model saved.')

    def load(self,model='model.json'):
        json_file = open(model,'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.fcn_model = model_from_json(loaded_model_json)
        self.fcn_model.load_weights(model.split('.')[0]+'.h5')
        print('Model loaded.')
