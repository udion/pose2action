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

|						Data 				    |	Classifier									|    Results  (Accuracy) (val%, train%)		|
|-----------------------------------------------|-----------------------------------------------|-------------------------------|
| Ground-Truth-Skeleton - 5 classes				|	Single LSTM, 3D coordinates					|	75.5%, 79.5%   		|
| Ground-Truth-Skeleton - 5 classes				|	2-Stacked LSTMs, 3D coordinates 			| 	77.1%, 80.4%  		|
| Ground-Truth-Skeleton - 5 classes				|	3-Stacked LSTMs, 3D coordinates 			| 	77.2%, 85.6%  		|
| Ground-Truth-Skeletons - 49 classes			|	2-Stacked LSTMs, 3D coordinates				|	59.7%, 72.5%		|
| Hourglass-Predicted-Skeletons - 8 classes		|	2-Stacked LSTMs, 3D coordinates				|	81.25% 						|

For the above mentioned 8 classes, some of the top accuracies models and their learning curve is shown below. Note that some of the models are not fully trained and will possibly score higher if training is completed.

| Classifier									|    Results  (Accuracy) (val%)		| training plots
|-----------------------------------------------|-----------------------------------------------|-------------------------------|
| 3D+2D projections LSTMS					|	82.527%   		|  ![test](./outputs/plots/inst_classifierX3.png) ![test](./outputs/plots/av_classifierX3.png) ![test](./outputs/plots/acc_classifierX3.png) |
| all 2D projections					|	64.233%   		|  ![test](./outputs/plots/inst_classifierX32d_all.png) ![test](./outputs/plots/av_classifierX32d_all.png) ![test](./outputs/plots/acc_classifierX32d_all.png) |

## Requirements and Instructions
Kindly use the requirements.txt to set up your machine for replicating this experiment. To train the models run `python LSTM_classifierX3cuda<one_of_model_names>.py` in the src folder. This will start the training for 50 epochs and keep saving the best and the last model so far along with the accuracy and loss results in `tr_models` and `outputs` respectively. 
