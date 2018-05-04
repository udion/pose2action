# pose2action

This is an attempt to classify the actions in the images and videos using the pose. For the purpose of this experiment
to get the poses from the the images and videos we are using the awesome repository @
![https://github.com/xingyizhou/pytorch-pose-hg-3d](https://github.com/xingyizhou/pytorch-pose-hg-3d)

This will hopefully help extend the pipeline of pose estimation to also perform classification. The current experiments are on the subset of
NTU video images dataset consisting of only 8 action classes.







SAHIL'S MODEL:
trainAcc: 99.34530490086045%

valAcc(testAcc): 81.25%

48->32 (fullyConnected)
32->2 x LSTM(hidden dimension = 160)
LSTM output = 160 -> 8 (fullyConnected) 

