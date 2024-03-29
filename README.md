# GM-TCNet: Gated Multi-scale Temporal Convolutional Network using Emotion Causality for Speech Emotion Recognition

# **_Note_**

We apologize for any inconvenience caused by some errors in the publication process. If you would like to read or follow our work, you can view it on arxiv or researchgate, or find the correct version of our paper on the GitHub named `GM-TCN_preprint.pdf` or [arxiv](https://arxiv.org/abs/2210.15834). 
If you have any questions, please feel free to contact me through **jiaxin-ye@foxmail.com**, and I will be honored to answer your questions. Thanks for your attention. Best wishes from JX Ye.

# Introduction

These are the **official repository** for the paper: JiaXin Ye#, Xin-Cheng Wen#, Xuan-Ze WANG, Yong Xu, Yan Luo, Chang-Li Wu, Li-Yan Chen, Kun-Hong Liu\*, GM-TCNet: Gated Multi-scale Temporal Convolutional Network using Emotion Causality for Speech Emotion Recognition, **Speech Communication (CCF-B)**. 

In this paper, we propose a Gated Multi-scale Temporal Convolutional Network (GM-TCNet) to construct a novel emotional causality representation learning component with a multi-scale receptive field. 

## Requirements

Our code is based on Python 3 (>= 3.8). There are a few dependencies to run the code. The major libraries are listed as follows:

* Tensorflow-gpu (== 2.4.0)
* Scikit-learn (== 1.0.2)
* NumPy (== 1.19.5)
* SciPy (== 1.8.0)
* librosa (==0.8.1)
* Pandas (== 1.4.1)

# Datasets

The four public emotion datasets are used in the experiments: the Institute of Automation of Chinese Academy of Sciences (CASIA), Berlin Emotional dataset (EMODB), Ryerson Audio-Visual dataset of Emotional Speech and Song (RAVDESS), and Surrey Audio-Visual Expressed Emotion dataset (SAVEE). The languages of both RAVDESS and SAVEE are English. While the EMODB and CASIA datasets contain German and Chinese speeches respectively.

# Features Processing

In the experiments, the 39-D MFCCs are extracted from the Librosa toolbox with the default settings. That is, the frame length is 0.05 s, the frame shift is 0.0125 s, the sample rate is 22050 Hz and the window function added for the speech is Hamming window. 

```python
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
```

# Implementation and Training

## Training GM-TCNet module

- ``python main.py``

## Default settings in GM-TCNet:

* batch size = 64, learning rate $\alpha$ = 0.001, epoch = 300
* Optimizer ='Adam', $\beta_1$ = 0.93, $\beta_2$ = 0.98, $\epsilon$ = 1e-8

# Folder structure

```
GM-TCNet
├─ Models
├─ Results
├─ Common_Model.py
├─ GMTCN_Model.py
├─ GTCM.py
├─ GM-TCN_preprint.pdf
├─ README.md
├─ Utils.py
└─ main.py
```

## Citation

If you compare or use our model in your own work, please cite the following
[paper](https://arxiv.org/abs/2210.15834):

```bibtex
@article{jiaxin2022GMTCN,
    title = {GM-TCNet: Gated Multi-scale Temporal Convolutional Network using Emotion Causality for Speech Emotion Recognition},
    author = {Jia-Xin Ye and Xin-Cheng Wen and Xuan-Ze Wang and Yong Xu and Yan Luo and Chang-Li Wu and Li-Yan Chen and Kun-Hong Liu},
    journal = {Speech Communication},
    volume = {145},
    pages = {21-35},
    year = {2022}
}
```
