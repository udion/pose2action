import sys
import torch
from opts import opts
import ref
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np
from functools import partial
import pickle
import os

###########################
path_to_train_labels = './sample_vids/train/labels'
path_to_val_labels = './sample_vids/val/labels'

train_data_path = '../../data/big_ntu_frames/train_data.pkl'
val_data_path = '../../data/big_ntu_frames/val_data.pkl'

keep_actions_train = range(49)
keep_actions_val = []
# usually they will be same
keep_actions_val = keep_actions_train
path_single_only_data = '../../data/big_ntu_frames/single_only'

try:
    os.makedirs(path_single_data)
except OSError:
    pass
###########################
train_labels = pd.read_csv(path_to_train_labels, header=None)
val_labels = pd.read_csv(path_to_val_labels, header=None)

with open(train_data_path, 'rb') as f:
    train_data = pickle.load(f)
with open(val_data_path, 'rb') as f:
    val_data = pickle.load(f)

sin_train_data = {}
sin_val_data = {}
###########################

print('processing the train_data ...')
for k in train_data:
    idx = train_labels[k, 1]
    if idx in keep_actions_train:
        sin_train_data[k] = train_data[k]
    else:
        pass
print('only single skeleton frames now, {} samples now'.format(len(sin_train_data)))
with open(path_single_only_data+'/train_data.pkl', 'wb') as handle:
	pickle.dump(sin_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('processing the val_data ...')
for k in val_data:
    idx = val_labels[k, 1]
    if idx in keep_actions_val:
        sin_val_data[k] = val_data[k]
    else:
        pass
print('only single skeleton frames now, {} samples now'.format(len(sin_val_data)))
with open(path_single_only_data+'/val_data.pkl', 'wb') as handle:
	pickle.dump(sin_val_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done')