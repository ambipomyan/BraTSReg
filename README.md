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

## sample run
```
inputs (HWC): 240 240 155
init blocks (HWC): 3 3 1
search window radius: 15
alpha: 1.0
----------------- Kid = 0 -----------------
dart_throw...
# of list points for mls: 4770
voxel dim by mm (HWC): 1 1 1
[  0.         268.11005203 250.48153625  19.13112647  25.98076211
 227.83107777  29.93325909  43.         242.99794238  44.93328388
  46.21688003 228.94540834 206.46065     48.61069841  38.48376281
  55.14526272 207.85331366 199.65219758  71.83313998  83.24662155
 196.36954957 113.46365057 113.09288218  75.59100476  85.49268975
 112.37882363 197.95454024 135.93380742 151.06621065 153.52849898
 144.58907289 100.64790112 180.42727067 155.25462956 170.10878872
 195.45076106 158.54967676 156.82155464 120.85114811 182.86880543
 159.53056134 164.37153038 163.64901466 165.08179791 173.05779381
 173.91377174 174.16371608 174.42763543 173.51368822 173.76420805]
h: 268.1100520308778
iter#: 0 F(Z): 0 iterDiff: 0 sw: 15
iter#: 0 F(Z): 0 iterDiff: 0 sw: 8
iter#: 0 F(Z): 0 iterDiff: 0 sw: 4
iter#: 0 F(Z): 0 iterDiff: 0 sw: 2
iter#: 0 F(Z): 0 iterDiff: 0 sw: 1
----------------- Kid = 1 -----------------
dart_throw...
# of list points for mls: 33
voxel dim by mm (HWC): 1 1 1
[  0.         144.87926008 148.49242405 100.62305899   0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.
   0.           0.           0.           0.           0.        ]
h: 148.49242404917499
iter#: 0 F(Z): 0 iterDiff: 0 sw: 15
iter#: 0 F(Z): 0 iterDiff: 0 sw: 8
iter#: 0 F(Z): 0 iterDiff: 0 sw: 4
iter#: 0 F(Z): 0 iterDiff: 0 sw: 2
iter#: 0 F(Z): 0 iterDiff: 0 sw: 1
```
