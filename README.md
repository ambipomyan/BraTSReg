# BraTSReg
Prepared for MICCAI 2022 challenge

This algorithm is designed based on QPDIR method, which is an intensity based image registration method for deformable registration.
This method is featured with its objective function formulation including similarity and regulation.
The similarity is based on individual voxel displacement and the regulation is inspired by the leave-one-out cross validation.
Considering that this optimization problem could be non-convex, non-linear, even non-continuous, the gradient-based methods may lead to a degrading of fine image information.
For implementation, gradient-free method based on block-matching and feature extraction is applied since the high spacial accuracy it could achieve.

For block-matching methods, given that it is an exhausted search, the displacement estimation can still be erroneous: for each independent estimation, only a small part of local image patch is involved, also, there is no interactions with the motion of neighbors. Thus, the outliers need to be filtered. 
