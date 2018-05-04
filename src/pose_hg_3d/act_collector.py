import os, shutil
import numpy as np
import pickle
import pandas as pd
import errno

############# specifics ###########
path_to_train_labels = './sample_vids/train/labels'
path_to_val_labels = './sample_vids/val/labels'

path_to_train_frames = './sample_vids/train_frames'
path_to_val_frames = './sample_vids/val_frames'

keep_actions_train = [0, 6, 12, 18, 24, 30, 36, 42]
keep_actions_val = []
# usually they will be same
keep_actions_val = keep_actions_train
path_small_data = './test_small'
###################################

try:
    os.makedirs(path_small_data+'/train_frames')
except OSError:
    pass
try:
    os.makedirs(path_small_data+'/val_frames')
except OSError:
    pass
####################################
train_labels = pd.read_csv(path_to_train_labels, header=None)
val_labels = pd.read_csv(path_to_val_labels, header=None)

train_dirs = os.listdir(path_to_train_frames)
val_dirs = os.listdir(path_to_val_frames)

def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

print('creating subset of training frames ...')
for act in train_dirs:
    act_id = int(act.replace('train', ''))
    act_id = train_labels.iloc[act_id, 1]
    if act_id in keep_actions_train:
        path_s = path_to_train_frames+'/'+act
        path_d = path_small_data+'/train_frames/'+act
        copy(path_s, path_d)
        print('{} copied at {}'.format(act, path_d))

print('creating subset of val frames ...')
for act in val_dirs:
    act_id = int(act.replace('val', ''))
    act_id = val_labels.iloc[act_id, 1]
    if act_id in keep_actions_val:
        path_s = path_to_val_frames+'/'+act
        path_d = path_small_data+'/val_frames/'+act
        copy(path_s, path_d)
        print('{} copied at {}'.format(act, path_d))