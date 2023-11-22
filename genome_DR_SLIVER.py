import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation,  Dropout, Input, concatenate, add, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from metrics import *
import keras.losses
keras.losses.dice_coef_loss=dice_coef_loss
import keras.metrics
keras.metrics.dice_coef=dice_coef
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal, VarianceScaling
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import adam,rmsprop,adadelta,adagrad
import tensorflow as tf


class ModelMGPU(Model):
    def __init__(self, model, gpus):
        pmodel = multi_gpu_model(model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
        
        
class GenomeHandler:

    """
    Defines the configuration and handles the conversion and mutation of
    individual genomes. Should be created and passed to a `DEvol` instance.

    ---
    Genomes are represented as fixed-with lists of integers corresponding
    to sequential layers and properties. 


    """

    def __init__(self, max_block_num,  max_filters,
                  input_shape,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None, learningrates=None, batch_size=None, augsize=None):
        """
        Creates a GenomeHandler according 

        Args:
            max_conv_layers: The maximum number of convolutional blocks          
            
                    layers, including output layer
            max_filters: The maximum number of conv filters (feature maps) in a
                    convolutional layer
            
            input_shape: The shape of the input
           
            batch_normalization (bool): whether the GP should include batch norm
            dropout : whether the GP should include dropout with probability of 0 to 0.6
            pooling (bool): whether the GP should include max pooling layer or average pooling
            optimizers (list): list of optimizers to be tried by the GP. By
                    default, the network uses Keras's built-in adam, rmsprop,
                    adagrad, and adadelta
            learning rate : list of learning rates. By default, 0.1, 0.01, 0.001
            batch size : list of batch size. By default, 8, 16, 32
            augmentation size: list of augmentation size. By default, 16000, 32000, 64000
            
            activations (list): list of activation functions to be tried by the
                    GP. By default, relu and sigmoid.
        Each convolutional blocks can contain maximum three convolutional layers.
                 
        """

        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.learningrate = learningrates or [
            0.1,
            0.01,
            0.001
        ]
        self.batchsize = batch_size or [
            
            8,
            16,
            32,

        ]
        self.augmentationsize = augsize or [
            16000,
            32000,
            64000,
            
            
        ]

        self.activation = activations or [
            'relu',
            'sigmoid',
        ]
        self.convolutional_layer_shape = [
            "active",
            "shortcon",
            # "typeshortcon",
            "longcon",
            # "typelongcon",
            "conv1",
            "conv size1",
            "conv2",
            "conv size2",
            "conv3",
            "conv size3",
            # "conv",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "pooling",
        ]

        self.layer_params = {

            "active": [0, 1],
            "shortcon": [0, 1],  # 0 is residual  1 is dense block
            # "typeshortcon": [0,1], # 0=elementwise sum, 1=concatenation
            "longcon": [0, 1],
            # "typelongcon":[0,1], # 0=elementwise sum, l=concatenation
            "conv1": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size1": [3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "conv2": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size2": [3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "conv3": [0, 1],
            # "conv": list(range(len(self.conv))),
            "conv size3": [3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "num filters": [2 ** i for i in range(3, filter_range_max)],
            # "num nodes": [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(7)],
            # "max pooling": list(range(3)) if max_pooling else 0,
            "pooling": [0, 1],  # 1=maxpooling, 0=averagepooling
        }
        

        self.convolution_layers = max_block_num
        self.convolution_layer_size = len(self.convolutional_layer_shape)

        self.input_shape = input_shape
        #
        

    def convParam(self, i):
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]
        


    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:  # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1
            if index == 98:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))


            if index == 99:

                genome[index] = np.random.choice(list(range(len(self.learningrate))))

            if index == 100:
                genome[index] = np.random.choice(list(range(len(self.batchsize))))
            if index == 101:
                genome[index] = np.random.choice(list(range(len(self.augmentationsize))))

        return genome

    def conv_block (self, m, dim, acti, bn, res, dp, cn1, cs1, cn2, cs2, cn3, cs3, active):

        if active and (cn1 or cn2 or cn3):
            init = VarianceScaling(scale=1.0 / 9.0)
            n = m
            n1 = K.zeros(tf.shape(m), dtype=m.dtype)
            n2 = K.zeros(tf.shape(m), dtype=m.dtype)
            n3 = K.zeros(tf.shape(m), dtype=m.dtype)
            # n1 = K.eval(n1)
            if cn1 == 1:
                n = Conv2D(dim, (cs1, cs1), activation=acti, padding='same', kernel_initializer=init)(m)
                n = BatchNormalization()(n) if bn else n
                n1 = n
            if cn2 == 1:
                n = Conv2D(dim, (cs2, cs2), activation=acti, padding='same', kernel_initializer=init)(n)
                n = BatchNormalization()(n) if bn else n
                n2 = n
            if cn3 == 1:
                n = Conv2D(dim, (cs3, cs3), activation=acti, padding='same', kernel_initializer=init)(n)
                n = BatchNormalization()(n) if bn else n
                n3 = n


            if res:  # if res==0 it is residual else dense block
                if cn1 == 0 and cn2 == 0 and cn3 == 1:
                    n = concatenate([n3, m], axis=3)
                elif cn1 == 0 and cn2 == 1 and cn3 == 0:
                    n = concatenate([n2, m], axis=3)
                elif cn1 == 0 and cn2 == 1 and cn3 == 1:
                    n = concatenate([n2,m] , axis=3)
                    n = Conv2D(dim, (cs3, cs3), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n3 = n
                    n = concatenate([n2, n3, m], axis=3)
                elif cn1 == 1 and cn2 == 0 and cn3 == 0:
                    n = concatenate([n1, m], axis=3)
                elif cn1 == 1 and cn2 == 0 and cn3 == 1:
                    n = concatenate([n1,  m], axis=3)
                    n = Conv2D(dim, (cs3, cs3), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n3 = n
                    n = concatenate([n1, n3, m], axis=3)
                elif cn1 == 1 and cn2 == 1 and cn3 == 0:
                    n = concatenate([n1,  m], axis=3)
                    n = Conv2D(dim, (cs2, cs2), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n2 = n
                    n = concatenate([n1, n2, m], axis=3)
                elif cn1 == 1 and cn2 == 1 and cn3 == 1:
                    n = concatenate([n1,  m], axis=3)
                    n = Conv2D(dim, (cs2, cs2), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n2 = n
                    n = concatenate([n1, n2, m], axis=3)
                    n = Conv2D(dim, (cs3, cs3), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    n3 = n
                    n = concatenate([n1, n2, n3, m], axis=3)

                n = Dropout(float(dp / 10.0))(n)
                return n

      

            else:
                # print(np.shape(n))
                # print(np.shape(m))
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  #  input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1), activation=acti, padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1), activation=acti, padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                # print(np.shape(n))
                # print(np.shape(m))
                n = keras.layers.Add()([n, m])
                n = Dropout(float(dp / 10.0))(n)
                return n
        else:
            return m
    

    def level_block(self, m, genome, depth, up, offset):

        
        if depth > 1:

            active = genome[offset]
            res = genome[offset + 1]  # short Connection
            # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
            lc = genome[offset + 2]  # long connection
            # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
            cn1 = genome[offset + 3]  # first conv layer
            cs1 = genome[offset + 4]  # first conv layer filter size
            cn2 = genome[offset + 5]  # second conv layer
            cs2 = genome[offset + 6]  # second conv layer filter size
            cn3 = genome[offset + 7]  # third conv layer
            cs3 = genome[offset + 8]  # third conv layer lize
            dim = genome[offset + 9]  # the number of filters
            bn = genome[offset + 10]  # the Batch normalization
            ac = genome[offset + 11]  # Activation functions
            dp = genome[offset + 12]  # the dropout
            pl = genome[offset + 13]  # type of pooling, maxpooling=1 or average pooling=0
            if ac == 1:
                acti = 'relu'
            else:
                acti = 'sigmoid'
            n= self.conv_block(m, dim, acti, bn, res, dp,  cn1 , cs1, cn2, cs2, cn3, cs3, active)
            if pl==1 and active and (cn1 or cn2 or cn3):
                m = MaxPooling2D()(n)
            elif pl==0 and active and (cn1 or cn2 or cn3):
                m = AveragePooling2D()(n)
            offset += self.convolution_layer_size
            # offset1=offset

            m = self.level_block( m, genome, depth-1, up, offset)

            if up :
                
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:

                
                offset -= self.convolution_layer_size
                active = genome[offset]
                if active and (cn1 or cn2 or cn3):
                    res = genome[offset + 1]  # short Connection
                    # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
                    lc = genome[offset + 2]  # long connection
                    # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
                    cn1 = genome[offset + 3]  # first conv layer
                    cs1 = genome[offset + 4]  # first conv layer filter size
                    cn2 = genome[offset + 5]  # second conv layer
                    cs2 = genome[offset + 6]  # second conv layer filter size
                    cn3 = genome[offset + 7]  # third conv layer
                    cs3 = genome[offset + 8]  # third conv layer lize
                    dim = genome[offset + 9]  # the number of filters
                    bn = genome[offset + 10]  # the Batch normalization
                    ac = genome[offset + 11]  # Activation functions
                    dp = genome[offset + 12]  # the dropout
                    pl = genome[offset + 13]  # type of pooling, maxpooling=1 or average pooling=0
                    if ac == 1:
                        acti = 'relu'
                    else:
                        acti = 'sigmoid'
                #dim= int(m.shape[-1])
                    m = Conv2DTranspose(dim, (3,3), strides=(2,2), activation=acti, padding='same')(m)

            
            #offset -= self.convolution_layer_size
            active=genome[offset]
            if active and (cn1 or cn2 or cn3):
                res = genome[offset + 1]  # short Connection
                # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
                lc = genome[offset + 2]  # long connection
                # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
                cn1 = genome[offset + 3]  # first conv layer
                cs1 = genome[offset + 4]  # first conv layer filter size
                cn2 = genome[offset + 5]  # second conv layer
                cs2 = genome[offset + 6]  # second conv layer filter size
                cn3 = genome[offset + 7]  # third conv layer
                cs3 = genome[offset + 8]  # third conv layer lize
                dim = genome[offset + 9]  # the number of filters
                bn = genome[offset + 10]  # the Batch normalization
                ac = genome[offset + 11]  # Activation functions
                dp = genome[offset + 12]  # the dropout
                pl = genome[offset + 13]  # type of pooling, maxpooling=1 or average pooling=0
                if ac == 1:
                    acti = 'relu'
                else:
                    acti = 'sigmoid'

                n = concatenate([n, m], axis=3) if lc==1  else m #keras.layers.Add()([n,m])
                m = self.conv_block(n, dim, acti, bn, res, dp, cn1, cs1, cn2, cs2, cn3, cs3, active)
                

        else:

            
            active=genome[offset]
            if active:
                res = genome[offset + 1]  # short Connection
                # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
                lc = genome[offset + 2]  # long connection
                # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
                cn1 = genome[offset + 3]  # first conv layer
                cs1 = genome[offset + 4]  # first conv layer filter size
                cn2 = genome[offset + 5]  # second conv layer
                cs2 = genome[offset + 6]  # second conv layer filter size
                cn3 = genome[offset + 7]  # third conv layer
                cs3 = genome[offset + 8]  # third conv layer lize
                dim = genome[offset + 9]  # the number of filters
                bn = genome[offset + 10]  # the Batch normalization
                ac = genome[offset + 11]  # Activation functions
                dp = genome[offset + 12]  # the dropout
                pl = genome[offset + 13]  # type of pooling, maxpooling=1 or average pooling=0
                if ac == 1:
                    acti = 'relu'
                else:
                    acti = 'sigmoid'
                m = self.conv_block(m, dim, acti, bn, res, dp, cn1, cs1, cn2, cs2, cn3, cs3, active)
                offset -= self.convolution_layer_size
            #m = self.conv_block(m, 32, 'relu', True, True, 10, 3, 3, 1)

        return m
    
    def EvoUNet(self, genome,  depth, upconv= False):
        out_ch = 1
        
        print(upconv)
        img_shape=(128,128,1)
        i = Input(shape=img_shape)
        o= self.level_block(i, genome, depth, upconv,0)
        o = Conv2D(out_ch, (1,1), activation='sigmoid')(o)
        return Model(inputs=i, outputs=o)

    def decode(self, genome):
        # if not self.is_compatible_genome(genome):
        #     raise ValueError("Invalid genome for specified configs")
        print(genome)
        model = self.EvoUNet(genome,  7, upconv=False)

        model.summary()
        pl_model = ModelMGPU(model, gpus=2)
        
        op = self.optimizer[genome[98]]
        batch = self.batchsize[genome[100]]
        aug = self.augmentationsize[genome[101]]
        print(op)
        if op=='adam':
            pl_model.compile(optimizer=adam(lr=self.learningrate[genome[99]]), loss= dice_coef_loss,
            metrics=[dice_coef])
        elif op=='rmsprop':
            pl_model.compile(optimizer=rmsprop(lr=self.learningrate[genome[99]]), loss=dice_coef_loss,
                          metrics=[dice_coef])
        elif op=='adadelta':

            pl_model.compile(optimizer=adadelta(lr=self.learningrate[genome[99]]), loss=dice_coef_loss,
                          metrics=[dice_coef])
        else:
            pl_model.compile(optimizer=adagrad(lr=self.learningrate[genome[99]]), loss=dice_coef_loss,
                          metrics=[dice_coef])

        return pl_model, model , batch, aug

    def genome_representation(self):
        encoding = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)

        encoding.append("Optimizer")
        encoding.append("Learning Rate")
        encoding.append("Batch Size")
        encoding.append("Augmentation Size")

        return encoding

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))

        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome.append(np.random.choice(list(range(len(self.learningrate)))))
        genome.append(np.random.choice(list(range(len(self.batchsize)))))
        genome.append(np.random.choice(list(range(len(self.augmentationsize)))))
        genome[0] = 1
        #genome[40] = 1
        print(genome)

        return genome



    def best_genome(self, csv_path, metric='accuracy', include_metrics=True):
        best = max if metric is 'accuracy' else min
        col = -1 if metric is 'accuracy' else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        
        return genome

    def decode_best(self, csv_path, metric='accuracy'):
        
        return self.decode(self.best_genome(csv_path, metric, False))
