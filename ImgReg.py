import os
import numpy as np
import math
import random

import nibabel as nib
from nibabel.testing import data_path

from BlockCoordinateDecent import throwDarts, kNN, mls
from QPDIR                 import computeFuncRes, updateDisplacementField, computeIterDiff
from ImgSeg                import createMask, saveImg, genFullPred, genPredImg
from eval                  import computeMAE, computeRobustness, computeJacobiDeterminant

# block matching settings for tests
BLOCKS  = 512
THREADS = 512
BUCKETS = 512

# ----- load data ----- #
original_file   = "/home/kyan2/Desktop/BraTSReg_Validation_Data/BraTSReg_147/BraTSReg_147_00_0000_t1.nii.gz"
following_file  = "/home/kyan2/Desktop/BraTSReg_Validation_Data/BraTSReg_147/BraTSReg_147_01_0399_t1.nii.gz"
landmark_file   = "/home/kyan2/Desktop/BraTSReg_Validation_Data/BraTSReg_147/BraTSReg_147_01_0399_landmarks.csv"

# find data path for original scan and fllowing scan
orignial  = os.path.join(data_path, original_file)
following = os.path.join(data_path, following_file)

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

n_slice = 0

# get image slices
#
n_slice = 74
C = 10
#
fixed_data  = np.zeros((C, H, W), dtype=int)
moving_data = np.zeros((C, H, W), dtype=int)
print("sliced input dims(HWC):", H, W, C)

for k in range(C):
    for i in range(H):
        for j in range(W):
            fixed_data[k][i][j]  = fixed_data_raw[i][j][n_slice + k]
            moving_data[k][i][j] = moving_data_raw[i][j][n_slice + k]

# create segmented mask image
mask_data = createMask(moving_data, H, W, C)

# saving images for visualization
print("saving images...")
saveImg(fixed_data,  H, W, C, "fixed_test.jpg" , 1)
saveImg(moving_data, H, W, C, "moving_test.jpg", 1)
saveImg(mask_data,   H, W, C, "mask_test.jpg"  , 1)

# ----- set parameters ----- #

# init block size
rx = 3
ry = 3
rz = 3
print("block radius(HWC):", rx, ry, rz)

# search window size (and penalty parameter mu)
# cubic window, sx == sy == sz
sx = 10
sy = 10
sz = 10
sw = sx
#mu = sw**2 / 2
print("init search window radius(HWC):", sx, sy, sz)

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
dpx = rx # larger numbers for quicker tests
dpy = ry
dpz = rz

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

    #print(np.amax(z_ws[0]), np.amax(z_ws[1]), np.amax(z_ws[2]))

    maxIter = 50
    SWin = sw
    while SWin != 0:
        mu = 1/SWin # use mu to replace 1/(2*mu**2)
        mu = mu**2
        for i in range(maxIter):
            # update obj function
            computeFuncRes(A, KNN, knn, b, x, r, p, Ap, Zold, Y, L, alpha, 2*mu)

            # update displacement field
            objVal, ccVal = updateDisplacementField(fixed_data, moving_data, F, I, z_ws, Z, Y, L, localVals, mu, sx, sy, sz, rx, ry, rz, H, W, C)

            # compute diff between iters
            nrmZ, nrmABS = computeIterDiff(Z, Zold, Y, L)

            print("iter#:", i, "F(Z):", objVal,                             \
                  "f(z):", ccVal, "||AX-Z||:", nrmABS, "||Xk+1-Xk||", nrmZ, \
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

    #print("i, d[0], d[1], d[2], d_ws[0], d_ws[1], d_ws[2], j, fixed, moving", i, d[0][i], d[1][i], d[2][i], d_ws[0][i], d_ws[1][i], d_ws[2][i], d_ws[2][i]*H*W + d_ws[0][i]*W + d_ws[1][i], \
                                                                              #fixed_data[d_ws[2][i]][d_ws[0][i]][d_ws[1][i]], moving_data[d_ws[2][i]][d_ws[0][i]][d_ws[1][i]])

with open('weights', 'w') as f:
    f.write("%s\n" % dL)
    for item in sol:
        f.write("%s\n" % item)

# ----- check reg result(s) ----- #
pred_darts = genFullPred(d, d_ws, L, moving_data, H, W, C, dpx, dpy, dpz)
pred_data  = genPredImg(pred_darts, moving_data, H, W, C)
saveImg(pred_data,  H, W, C, "pred_test.jpg", 1)

# ----- check similarity metrics ----- #
print("===========")
print("evaluation:")
# -- MAE -- #
res_before_mae = computeMAE(moving_data, fixed_data, H, W, C)
res_after_mae  = computeMAE(pred_data,   fixed_data, H, W, C)
print("MAE: before:", res_before_mae, "after:", res_after_mae)

# -- Robustness --#
r, n = computeRobustness(moving_data, fixed_data, pred_data, landmark_file)
print("Robustness:", r, "/", n)

# -- Jacobian Determinat -- #
n_negative, JD, jd_max, jd_mean = computeJacobiDeterminant(d, d_ws, L, H, W, C, dpx, dpy, dpz, "jd_test.jpg")
print("Jacobian Determinat: #negative elements: ", n_negative, "/", H*W*C, "Max and Mean vals:", jd_max, jd_mean)
print("===========")
# ----- Last Line ----- #
