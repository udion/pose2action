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

#opt.demo to point to the dir containing train_frames and val_frames dir

def main():
	pickle.load = partial(pickle.load, encoding="latin1")
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
	opt = opts().parse()
	if opt.loadModel != 'none':
		model = torch.load(opt.loadModel).cuda()	
	else:
		model = torch.load('../../models/hgreg-3d.pth').cuda()
	action_dirs_tr = os.listdir(opt.demo+'/train_frames')
	action_dirs_vl = os.listdir(opt.demo+'/val_frames')
	
	my_tr_dict = {}
	for a_dir in action_dirs_tr:
		all_frames = os.listdir('{}/{}'.format(opt.demo+'/train_frames/', a_dir))
		n_frames = len(all_frames)
		frames_seq = np.zeros((n_frames, 16, 3))	
		for idx, frame in enumerate(all_frames):
			img = cv2.imread('{}/{}/{}'.format(opt.demo+'/train_frames/',a_dir, frame))
			input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
			input = input.view(1, input.size(0), input.size(1), input.size(2))
			input_var = torch.autograd.Variable(input).float().cuda()
			output = model(input_var)
			pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
			reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
			#point_3d are the 3 dimensional co-ordinates of the 16 joints
			point_3d = np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1)
			frames_seq[idx, :, :] = point_3d
		my_tr_dict[a_dir] = frames_seq
		print('{} done!'.format(a_dir))
	with open(opt.demo+'/train_data.pkl', 'wb') as handle:
		pickle.dump(my_tr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	my_vl_dict = {}
	for a_dir in action_dirs_vl:
		all_frames = os.listdir('{}/{}'.format(opt.demo+'/val_frames/', a_dir))
		n_frames = len(all_frames)
		frames_seq = np.zeros((n_frames, 16, 3))	
		for idx, frame in enumerate(all_frames):
			img = cv2.imread('{}/{}/{}'.format(opt.demo+'/val_frames/',a_dir, frame))
			input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
			input = input.view(1, input.size(0), input.size(1), input.size(2))
			input_var = torch.autograd.Variable(input).float().cuda()
			output = model(input_var)
			pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
			reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
			#point_3d are the 3 dimensional co-ordinates of the 16 joints
			point_3d = np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1)
			frames_seq[idx, :, :] = point_3d
		my_vl_dict[a_dir] = frames_seq
		print('{} done!'.format(a_dir))
	with open(opt.demo+'/val_data.pkl', 'wb') as handle:
		pickle.dump(my_vl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()