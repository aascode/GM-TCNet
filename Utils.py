from Config import Config
from DTCNs import TCN
from typing import Tuple
import sys
import matplotlib.pyplot as plt
import numpy as np
import os 
#import pyaudio
from tqdm import tqdm

#平均信号的长度
mean_signal_length = 96000
EMO_LABEL = ("angry", "boredom", "disgust", "fear", "happy", "neutral","sad")
CASIA_LABEL = ("angry", "fear", "happy", "neutral","sad","surprise")
SAVEE_LABELS =("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")
RAVDESS_LABEL = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")

def get_all_data(data_path: str, mfcc_len: int = 39, class_labels: Tuple = ("angry", "fear", "happy", "neutral", "sad", "surprise"), flatten: bool = False):
    x = []
    y = []
    current_dir =  os.getcwd() #返回当前的进程目录
    sys.stderr.write('当前的进程目录是: %s\n' % current_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("开始读取文件夹 %s\n" % directory)
        os.chdir(directory)
        for filename in tqdm(os.listdir('.')):
            if not filename.endswith('.csv'):
                continue
            filepath = os.getcwd() + '/' + filename
            feature_vector = np.loadtxt(filepath, delimiter=",", dtype = np.float32, encoding="gbk")
            x.append(feature_vector)
            y.append(i)
        sys.stderr.write("结束读取文件夹 %s\n" %directory)
        os.chdir('..')
    os.chdir(current_dir)
    return np.array(x), np.array(y)


def get_feature(file_path: str, mfcc_len: int = 39, flatten: bool = False):
    signal, fs = librosa.load(file_path)
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature

def get_mixed_feature(data_path: str, mfcc_len: int = 39, class_labels: Tuple = ("angry", "fear", "happy", "neutral","sad","surprise"), flatten: bool = False):
    """
        Input：音频所在文件夹
        Output：CSV格式的MFCC特征
    """
    class_labels = CASIA_LABEL
    current_dir =  os.getcwd() #返回当前的进程目录
    if not os.path.exists(csv_train):
        print(csv_train+"文件夹创建成功")
        os.makedirs(csv_train)
        os.chdir(csv_train)
    else:
        os.chdir(csv_train)
    for i, directory in enumerate(class_labels):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory+"文件夹创建成功")
    os.chdir('..')
    datapath = []
    labels = []
    sys.stderr.write('当前的进程目录是: %s\n' % current_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("开始读取文件夹 %s\n" % directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chdir(directory)
            print(directory+"文件夹创建成功")
        else:
            os.chdir(directory)
        for filename in tqdm(os.listdir('.')):
            if not filename.endswith('wav'):
                continue
            filepath = os.getcwd() + '/' + filename
            datapath.append(filepath)
            labels.append(i)
        sys.stderr.write("结束读取文件夹 %s\n" %directory)
        os.chdir('..')
    os.chdir(current_dir)
    for video_path,label in tqdm(zip(datapath,labels)):
        filename = video_path[video_path.rfind('/')+1:-4]
        feature_vector = get_feature(file_path = video_path, mfcc_len = mfcc_len, flatten = flatten)
        np.savetxt(csv_train+"/" + class_labels[label] +"/" +filename+'_raw'+'.csv', feature_vector, delimiter = ',')