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

def main():
	pickle.load = partial(pickle.load, encoding="latin1")
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
	opt = opts().parse()
	if opt.loadModel != 'none':
		model = torch.load(opt.loadModel).cuda()	
	else:
		model = torch.load('../../tr_models/hgreg-3d.pth').cuda()

	#opt.demo has the path to dir containing frames of demo video
	all_frames = os.listdir(opt.demo)
	n_frames = len(all_frames)
	#specifics
	dir_name = opt.demo.split('/')[-1]
	save_path = '../../output/demo/'+dir_name
	try:
		os.makedirs(save_path)
	except OSError:
		pass

	for idx, frame in enumerate(all_frames):
		print('processing frame {}'.format(idx))
		img = cv2.imread(opt.demo+'/'+frame)
		input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
		input = input.view(1, input.size(0), input.size(1), input.size(2))
		input_var = torch.autograd.Variable(input).float().cuda()
		output = model(input_var)
		pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
		reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
		debugger = Debugger()
		debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
		debugger.addPoint2D(pred, (255, 0, 0))
		debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
		# debugger.showImg(pause = True)
		debugger.saveImg(path=save_path+'/frame{}.jpg'.format(idx))
		debugger.save3D(path=save_path+'/frame_p3d{}.jpg'.format(idx))
		print('frame {} done'.format(idx))

if __name__ == '__main__':
	main()
