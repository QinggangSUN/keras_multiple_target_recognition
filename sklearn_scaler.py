# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 20:30:52 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import preprocessing


def std_scaler(raw_data):#input 3d list [set][frame][framelength]
    d1 = [len(srci) for srci in raw_data] #number of frames in each set
    all_data =  np.vstack(raw_data)
    scaler = preprocessing.StandardScaler()
    std_data = scaler.fit_transform(all_data)

    data_sets = []
    si = 0
    for di in d1:
        data_sets.append(std_data[si:si+di].tolist())
        si+=di

    return data_sets, scaler

def max_abs_scaler(raw_data):#input 3d list [set][frame][framelength]
    d1 = [len(srci) for srci in raw_data] #number of frames in each set
    all_data =  np.vstack(raw_data)
    scaler = preprocessing.MaxAbsScaler() #to [-1,1]
    std_data = scaler.fit_transform(all_data)

    data_sets = []
    si = 0
    for di in d1:
        data_sets.append(std_data[si:si+di].tolist())
        si+=di

    return data_sets, scaler

def min_max_scaler(raw_data):#input 3d list [set][frame][framelength]
    d1 = [len(srci) for srci in raw_data] #number of frames in each set
    all_data =  np.vstack(raw_data)
    scaler = preprocessing.MinMaxScaler() #to [0,1]
    std_data = scaler.fit_transform(all_data)

    data_sets = []
    si = 0
    for di in d1:
        data_sets.append(std_data[si:si+di].tolist())
        si+=di

    return data_sets, scaler