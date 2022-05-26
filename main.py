# -*- coding:UTF-8 -*-
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from Utils import  get_all_data
import datetime
import matplotlib.pyplot as pl
from sklearn import metrics
from tensorflow.keras import Sequential, Model, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from Config import Config
import matplotlib.pyplot as plt

from GMTCN_Model import GMTCN_Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
session = tf.compat.v1.Session(config=config) # 设置session KTF.set_session(sess)
print(gpus)

# CLASS LABELS
casia_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#casia
emodb_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#emodb
savee_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#savee
ravde_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
# Data Paths
DATA_PATH_casia = 'CASIA_PATH'
DATA_PATH_emodb = 'EMODB_PATH'
DATA_PATH_savee = 'SAVEE_PATH'
DATA_PATH_ravde = 'RAVDESS_PATH'
# MFCC Feature Path
x_casia, y_casia= get_all_data(DATA_PATH_casia, class_labels = casia_CLASS_LABELS, flatten = False)
x_emodb, y_emodb= get_all_data(DATA_PATH_emodb, class_labels = emodb_CLASS_LABELS, flatten = False)
x_savee, y_savee= get_all_data(DATA_PATH_savee, class_labels = savee_CLASS_LABELS, flatten = False)
x_ravde, y_ravde= get_all_data(DATA_PATH_ravde, class_labels = ravde_CLASS_LABELS, flatten = False)
# Data Dict
data = {"CASIA":(x_casia, y_casia, casia_CLASS_LABELS),
"EMODB":(x_emodb, y_emodb, emodb_CLASS_LABELS),
"SAVEE":(x_savee, y_savee, savee_CLASS_LABELS),
"RAVDE":(x_ravde, y_ravde, ravde_CLASS_LABELS)}
data_name = ["CASIA","EMODB","SAVEE","RAVDE"]

source_name = 'CASIA'
CLASS_LABELS = data[source_name][2]
x_source = data[source_name][0]
y_source = data[source_name][1]
y_source = to_categorical(y_source,num_classes=len(CLASS_LABELS))
model = GMTCN_Model(input_shape = (x_source.shape[1:]), class_label = CLASS_LABELS)
model.train(x_source, y_source,None,None,n_epochs=200,data_name = source_name)
