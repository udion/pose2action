import cv2
import os
import shutil

try:
	os.mkdir('train_frames')
except OSError:
	pass
try:
	os.mkdir('val_frames')
except OSError:
	pass

train_files = os.listdir('./train')
print('processing train folder ...')
for file in train_files:
	if 'train' in file:
		vidcap = cv2.VideoCapture('./train/{}'.format(file))
		folder_name = file.replace('.avi', '')
		try:
			os.mkdir('./train_frames/{}'.format(folder_name))
		except OSError:
			pass
		success,image = vidcap.read()
		count = 0
		success = True
		while success:
			resized_image = cv2.resize(image, (256, 256))
			cv2.imwrite('./train_frames/{}/frame{}.jpg'.format(folder_name, count), resized_image)     # save frame as JPEG file      
			success,image = vidcap.read()
			# print('Read a new frame: ', success)
			count += 1
		print('./train_frames/{} done!'.format(folder_name))

val_files = os.listdir('./val')
print('processing val folder ...')
for file in val_files:
	if 'val' in file:
		vidcap = cv2.VideoCapture('./val/{}'.format(file))
		folder_name = file.replace('.avi', '')
		try:
			os.mkdir('./val_frames/{}'.format(folder_name))
		except OSError:
			pass
		success,image = vidcap.read()
		count = 0
		success = True
		while success:
			resized_image = cv2.resize(image, (256, 256))
			cv2.imwrite('./val_frames/{}/frame{}.jpg'.format(folder_name, count), resized_image)     # save frame as JPEG file      
			success,image = vidcap.read()
			# print('Read a new frame: ', success)
			count += 1
		print('./val_frames/{} done!'.format(folder_name))