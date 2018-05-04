# Action-Recognition

Course Project for CS763 : Computer Vision Course, IIT Bombay

## Abstract

### Problem Statement
Given a Video containing Human body Motion you have to recognize the action agent is performing.

### Solution Approaches
We started with Action Recognition from skeleton estimates of Human Body. 
Given 3D ground truth coordinates of Human Body (obtained from Kinect Cameras) we tried to use LSTMS as well as Temporal Convolutions for learning skeleton representation of Human Activity Recognition.
We also tried fancier LSTMs as well where we projected the 3D coordinates onto x-y plane, y-z plane, z-x plane followed by 1D convolutions and subsequently adding the outputs of the 4 LSTMs (x-y, y-z, z-x, 3D). Additionally we tried variants where we chose three out of the four LSTMs and compared performance among different projections.
Then we moved to Action Recognition from Videos.
We used pretrained Hourglass Network to estimate joints at each frame in videos and used similar LSTMs to perform the task of Action Recognition.

## Dependencies
numpy==1.11.0
pandas==0.17.1
matplotlib==1.5.1
keras==2.1.6
torch==0.4.0

## Instructions

## Results

|						Data 				    |	Classifier									|    Results  (Accuracy)		|
|-----------------------------------------------|-----------------------------------------------|-------------------------------|
| Ground-Truth-Skeleton - 5 classes				|	Single LSTM, 3D coordinates					|	75.5%, 79.5% (train)  		|
| Ground-Truth-Skeleton - 5 classes				|	2-Stacked LSTMs, 3D coordinates 			| 	77.1%, 80.4% (train)  		|
| Ground-Truth-Skeleton - 5 classes				|	3-Stacked LSTMs, 3D coordinates 			| 	77.2%, 85.6% (train)  		|
| Ground-Truth-Skeletons - 49 classes			|	2-Stacked LSTMs, 3D coordinates				|	59.7%, 72.5% (train)		|
| Hourglass-Predicted-Skeletons - 8 classes		|	2-Stacked LSTMs, 3D coordinates				|	81.25% 						|
| Hourglass-Predicted-Skeletons - 8 classes 	|	4 LSTMS, outputs fused after 1D conv        |	############## 	 			|
| Hourglass-Predicted-Skeletons - 49 classes 	|	4 LSTMS, outputs fused after 1D conv        |	############## 	 			|


## Intro
This is an attempt to classify the actions in the images and videos using the pose. For the purpose of this experiment
to get the poses from the the images and videos we are using the awesome repository @
[https://github.com/xingyizhou/pytorch-pose-hg-3d](https://github.com/xingyizhou/pytorch-pose-hg-3d)

This will hopefully help extend the pipeline of pose estimation to also perform classification. The current experiments are on the subset of
NTU video images dataset consisting of only 8 action classes.

The subset of the NTU dataset used for this project is the following

| Action            | label Id      |
| -------------     |:-------------:|
| drink water       | 0             |
| throw             | 1             |
| tear up paper     | 2             |
| take off glasses  | 3             |
| put something inside pocket / take out something from pocket | 4             |
| pointing to something with finger | 5             |
| wipe face | 6             |
| falling | 7             |

## Pipeline
The input is a sequence of frames (i.e video) which first passes through a trained model [available here](https://github.com/xingyizhou/pytorch-pose-hg-3d).
This produces the estmates for the pose in 3D, this 3D pose passes through our network (which takes it various projections) and is used as the main features to classify the action from the above 8 categories.

<p align='center'>
  <img src='./outputs/readme_out/input.gif' alt='input'/>
</p>

<p align='center'>
  <img src='./outputs/readme_out/xingy_net.png' alt='x net'/>
</p>

| 2d            | 3d      |
| -------------     |:-------------:|
| ![input](./outputs/readme_out/output_ske.gif)      | ![input](./outputs/readme_out/3d_ske.gif)           |
   


<p align='center'>
  <img src='./outputs/readme_out/main_model0.png' alt='main model0' style="width: 1200px; height: 900px" />
</p>

`predicted action : tear up paper`
(check the load_testbed in notebook to verify this example)

We also tried many different variations for our classifier model, which includes simple 2 layered LSTM network, another type of variation included LSTM models based on only some of the 2d projection of the pose (say XY or YZ or ZX) etc.

## Some Results




SAHIL'S MODEL:
trainAcc: 99.34530490086045%

valAcc(testAcc): 81.25%

48->32 (fullyConnected)
32->2 x LSTM(hidden dimension = 160)
LSTM output = 160 -> 8 (fullyConnected) 

