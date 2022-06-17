# BraTSReg
Prepared for MICCAI 2022 challenge

## introduction
This algorithm is designed based on QPDIR method, which is an intensity based image registration method for deformable registration.
This method is featured with its objective function formulation including similarity and regulation.
The similarity is based on individual voxel displacement and the regulation is inspired by the leave-one-out cross validation.
Considering that this optimization problem could be non-convex, non-linear, even non-continuous, the gradient-based methods may lead to a degrading of fine image information.
For implementation, gradient-free method based on block-matching and feature extraction is applied since the high spacial accuracy it could achieve.

For block-matching methods, given that it is an exhausted search, the displacement estimation can still be erroneous: for each independent estimation, only a small part of local image patch is involved, also, there is no interactions with the motion of neighbors. Thus, the outliers need to be filtered. 

## dependency
python packages:
numpy
nibabel

# sample run
inputs (HWC): 240 240 155
init blocks (HWC): 3 3 1
search window radius: 15
alpha: 1.0
voxel dim by mm (HWC): 1 1 1
----------------- Kid = 0 -----------------
dart_throw...
# of list points for mls: 100
iter#: 0 F(Z): 0 iterDiff: 0 sw: 15
iter#: 0 F(Z): 0 iterDiff: 0 sw: 8
iter#: 0 F(Z): 0 iterDiff: 0 sw: 4
iter#: 0 F(Z): 0 iterDiff: 0 sw: 2
iter#: 0 F(Z): 0 iterDiff: 0 sw: 1
----------------- Kid = 1 -----------------
dart_throw...
# of list points for mls: 100
iter#: 0 F(Z): 0 iterDiff: 0 sw: 15
iter#: 0 F(Z): 0 iterDiff: 0 sw: 8
iter#: 0 F(Z): 0 iterDiff: 0 sw: 4
iter#: 0 F(Z): 0 iterDiff: 0 sw: 2
iter#: 0 F(Z): 0 iterDiff: 0 sw: 1

