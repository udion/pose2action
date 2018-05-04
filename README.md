# pose2action

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
  <img src='./outputs/readme_out/main_model0.png' alt='main model0' style="width: 1000px; height: 1000px" />
</p>






SAHIL'S MODEL:
trainAcc: 99.34530490086045%

valAcc(testAcc): 81.25%

48->32 (fullyConnected)
32->2 x LSTM(hidden dimension = 160)
LSTM output = 160 -> 8 (fullyConnected) 

