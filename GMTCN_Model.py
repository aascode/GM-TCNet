import numpy as np
import tensorflow.keras.backend as K
import sys
import os
from numpy.random import seed
seed(1024)
import tensorflow as tf
tf.random.set_seed(2048)
import tensorflow.keras.layers
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import optimizers,callbacks
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import Masking,Dropout,Dense,Activation,Embedding,Input,LSTM,GRU,Bidirectional,GlobalAveragePooling1D,GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape,Flatten,BatchNormalization,Bidirectional
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.regularizers import l1
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping,History,ModelCheckpoint
from typing import List, Tuple
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

import datetime
from Common_Model import Common_Model
from GTCM import *


class GMTCN_Model(Common_Model):
    def __init__(self, input_shape,class_label,**params):
        super(GMTCN_Model,self).__init__(**params)
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.batch_size = 64
        self.n_epochs = 300
        self.n_split = 10
        self.random_state_k = 32

    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1],))
        self.tcn = GTCM(nb_filters=39,
                               kernel_size=2, 
                               nb_stacks=1, #增加堆栈块会增加感受野大小
                               dilations=[2 ** i for i in range(7)],
                               activation='relu',
                               use_skip_connections=True, 
                               dropout_rate=0.0,
                               return_sequences=True, 
                               name='GTCM')(self.inputs)
        self.Conv1D = GlobalAveragePooling1D()(self.tcn)
        self.predictions = Dense(self.num_classes,activation='softmax')(self.Conv1D)
        self.model = Model(inputs = self.inputs,outputs = self.predictions)
        self.model.compile(loss = 'categorical_crossentropy', 
                           optimizer = Adam(learning_rate=0.001,beta_1=0.93,beta_2=0.98,epsilon=1e-8), 
                           metrics = ['accuracy'])
    
    def train(self, x, y, x_test, y_test, n_epochs = 100,data_name = None, fold = None , random = None):
        filepath='./Models/'# training model folder
        resultpath = './Results/'# evaluating result folder
        if n_epochs is not None:
            self.n_epochs = n_epochs
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        self.create_model()
        print(self.model.summary())

        i=1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now,'%Y-%m-%d %H:%M:%S')
        kfold = KFold(n_splits=self.n_split, shuffle=True, random_state=self.random_state_k)
        avg_accuracy = 0
        avg_loss = 0
        for train, test in kfold.split(x, y):
            self.create_model()
            folder_address = filepath+data_name+"_"+str(self.random_state_k)+"_"+now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path=folder_address+'/'+str(self.n_split)+"-fold_weights_best_"+str(i)+".hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1,save_weights_only=True,save_best_only=True,mode='max')
            max_acc = 0
            best_eva_list = []
            self.model.fit(x[train], y[train],validation_data=(x[test],  y[test]),batch_size = self.batch_size,epochs = self.n_epochs,verbose=1,callbacks=[checkpoint])
            self.model.load_weights(weight_path)#+source_name+'_single_best.hdf5')
            best_eva_list = self.model.evaluate(x[test],  y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))
        print("Average ACC:",avg_accuracy/self.n_split)
        self.acc = avg_accuracy/self.n_split
        writer = pd.ExcelWriter(resultpath+data_name+'_10fold_'+str(round(self.acc*10000)/100)+"_"+str(self.random_state_k)+"_"+now_time+'.xlsx')
        for i,item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j,l in enumerate(item):
                temp[self.class_label[j]]=item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer,sheet_name=str(i), encoding='utf8')

            df = pd.DataFrame(self.eva_matrix[i]).transpose()
            df.to_excel(writer,sheet_name=str(i)+"_evaluate", encoding='utf8')
            # plot_matrix(item,title='10fold_evaluate_'+str(i+1),axis_labels=self.class_label,thresh=0.5)
        writer.save()
        writer.close()
        
        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True