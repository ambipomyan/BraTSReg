import os
import numpy as np
import math
import random

import nibabel as nib
from nibabel.testing import data_path

from BlockCoordinateDecent import throwDarts, kNN, mls
from QPDIR import computeFuncRes, updateDisplacementField, computeIterDiff
from ImgSeg import convert2Int8, createMask, saveImg

# block matching settings for tests
BLOCKS  = 512
THREADS = 256
BUCKETS = 512

# ----- load data ----- #

# find data path for original scan and fllowing scan
orignial  = os.path.join(data_path, '/home/kyan2/Desktop/BraTSReg/BraTSReg_001_00_0000_t1.nii.gz')
following = os.path.join(data_path, '/home/kyan2/Desktop/BraTSReg/BraTSReg_001_01_0106_t1.nii.gz')

# load image
fixed_img   = nib.load(orignial)
moving_img  = nib.load(following)
fixed_data_raw  = fixed_img.get_fdata()
moving_data_raw = moving_img.get_fdata()

# check image shape
H = fixed_data_raw.shape[0]
W = fixed_data_raw.shape[1]
C = fixed_data_raw.shape[2]
print("input dims(HWC):", H, W, C)

# get image slices
#############
n_slice = 100
#############
C = 5
######
fixed_data  = np.zeros((C, H, W), dtype=int)
moving_data = np.zeros((C, H, W), dtype=int)
print("sliced input dims(HWC):", H, W, C)

for k in range(C):
    for i in range(H):
        for j in range(W):
            fixed_data[k][i][j]  = fixed_data_raw[i][j][n_slice + k]
            moving_data[k][i][j] = moving_data_raw[i][j][n_slice + k]

# create input image pairs and mask
fixed_data  = convert2Int8(fixed_data, H, W, C)
moving_data = convert2Int8(moving_data, H, W, C)
mask_data   = createMask(moving_data, H, W, C)

# saving images for visualization
print("saving images...")
saveImg(fixed_data,  H, W, "fixed.jpg" , 1)
saveImg(moving_data, H, W, "moving.jpg", 1)
saveImg(mask_data,   H, W, "mask.jpg"  , 100)

# check image type
res = (fixed_img.get_data_dtype() == np.dtype(np.int8))
if res == True: print("dtype: int8")

res = (fixed_img.get_data_dtype() == np.dtype(np.int16))
if res == True: print("dtype: int16")

# ----- set parameters ----- #

# init block size
rx = 3
ry = 3
rz = 1
print("block radius(HWC):", rx, ry, rz)

# search window size (and penalty parameter mu)
sw = 15 # 15x15x15 window
#mu = sw**2 / 2
print("init search window radius(HWC):", sw, sw, sw)

# regularization parameter
alpha = 1.0
print("alpha:", alpha)

# voxel dims
xmm = 1
ymm = 1
zmm = 1

# k-NN: number of connected componenets and max size, needs to be tuned
K    = 2
maxL = 500000
knn  = 50

# point cloud spacing for dart throw, needs to be tuned
dpx = 3 # larger numbers for quicker tests
dpy = 3
dpz = 1

# ------ set memory ------ #

# mask image
mask_data = mask_data.reshape(H*W*C)

# displacement field d and auxiliary variables z
d    = np.zeros((3, H*W*C), dtype=int)
Z    = np.zeros((3, maxL),  dtype=int)
Zold = np.zeros((3, maxL),  dtype=int)

# workspaces for d and z
d_ws = np.zeros((3, H*W*C), dtype=int)
z_ws = np.zeros((3, maxL),  dtype=int)

# matrices for mls
A   = np.zeros(knn * maxL)
KNN = np.zeros(knn * maxL, dtype=int)

# CG vectors
b  = np.zeros(maxL)
x  = np.zeros(maxL)
r  = np.zeros(maxL)
p  = np.zeros(maxL)
Ap = np.zeros(maxL)

# QP variables
Y = np.zeros((3, maxL))

# obj function res
F = np.zeros((2, BLOCKS*THREADS))
I = np.zeros(BLOCKS*THREADS, dtype=int)

# localSUM
localVals = np.zeros((2, BLOCKS))

# solution counter for d
dL = 0

# ----- run algorithm ----- #

for Kid in range(1, K+1):
    print("----------------- Kid =", Kid, "-----------------")

    # throwDarts
    L = throwDarts(mask_data, z_ws, dpx, dpy, dpz, H, W, C, Kid)

    # init guess of displacement field: 0
    for i in range(L):
        Z[0][i] = 0
        Z[1][i] = 0
        Z[2][i] = 0

        Zold[0][i] = Z[0][i]
        Zold[1][i] = Z[1][i]
        Zold[2][i] = Z[2][i]

    # kNN
    kNN(z_ws, L, z_ws, L, KNN, knn, xmm, ymm, zmm)

    # mls
    mls(z_ws, L, KNN, A, knn, xmm, ymm,zmm)

    maxIter = 10
    SWin = sw
    while SWin != 0:
        mu = SWin**2 / 2
        for i in range(maxIter):
            # update obj function
            computeFuncRes(A, KNN, knn, b, x, r, p, Ap, Zold, Y, L, alpha, mu)

            # update displacement field
            objVal, ccVal = updateDisplacementField(fixed_data, moving_data, F, I, z_ws, Z, Y, L, localVals, mu, SWin, SWin, SWin, rx, ry, rz)

            # compute diff between iters
            nrmZ, nrmABS = computeIterDiff(Z, Zold, Y, L)

            print("iter#:", i, "F(Z):", objVal, \
                  "f(z):", ccVal, "||AX-Z||:", nrmZ, "||Xk+1-Xk||", nrmABS, \
                  "sw:", SWin)

            if nrmZ == 0: break

        if SWin == 1: break
        SWin = int(round(SWin/2))

    # store solution to d
    for i in range(L):
        d[0][i+dL] = Z[0][i]
        d[1][i+dL] = Z[1][i]
        d[2][i+dL] = Z[2][i]

        d_ws[0][i+dL] = z_ws[0][i]
        d_ws[1][i+dL] = z_ws[1][i]
        d_ws[2][i+dL] = z_ws[2][i]

    dL += L


# ----- write displacement field to file ----- #

sol = np.zeros(6*dL)
for i in range(dL):
    sol[i]        = d_ws[0][i]
    sol[i + 1*dL] = d_ws[1][i]
    sol[i + 2*dL] = d_ws[2][i]

    sol[i + 3*dL] = d[0][i]
    sol[i + 4*dL] = d[1][i]
    sol[i + 5*dL] = d[2][i]

with open('weights', 'w') as f:
    f.write("%s\n" % dL)
    for item in sol:
        f.write("%s\n" % item)
