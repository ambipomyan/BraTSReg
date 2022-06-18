import numpy as np
import math
import random

from utils import distance

#----- dart throw -----#

def dart_throw(mask, S, dpx, dpy, dpz, H, W, C, Kid):
    # get tmp mask and voxel list sizes
    M = H*W*C
    S_tmp = np.zeros((3,M), dtype=int)
    N = getMaskVoxels(mask, S_tmp, H, W, C, Kid)
    # init tmp mask
    mask_tmp = np.zeros(M, dtype=int)
    # init voxel list
    for i in range(N):
        idx = H*W*S_tmp[2][i] + S_tmp[0][i]*W + S_tmp[1][i]
        mask_tmp[idx] = 1
    
    print("dart_throw...")
    count = 0
    M = N
    numDarts = 10000
    while M != 0:
        for n in range(numDarts):
            idx = random.randint(0, M)
            flg = hit(mask_tmp, dpx, dpy, dpz, S_tmp[0][idx], S_tmp[1][idx], S_tmp[2][idx], H, W, C)
            #print("is hit:", flg)
            if flg == 1:
               S[0][count] = S_tmp[0][idx]
               S[1][count] = S_tmp[1][idx]
               S[2][count] = S_tmp[2][idx]
               count += 1
        M = updateList(mask_tmp, S_tmp, M, H, W, C)
        #print("M:", M, "count:", count)

    return count

def getMaskVoxels(mask, S, H, W, C, Kid):
    count = 0
    N = 0
    for i in range(H):
        for j in range(W):
            for k in range(C):
                v = mask[count]
                if v == Kid:
                    S[0][N] = i
                    S[1][N] = j
                    S[2][N] = k
                    N += 1
                count += 1

    return N

def hit(mask, dpx, dpy, dpz, i, j, k, H, W, C):
    if mask[H*W*k + i*W + j] == 0: return 0
    
    # bounds for indeces
    i1 = i - dpx
    j1 = j - dpy
    k1 = k - dpz
    if i1 < 0: i1 = 0
    if j1 < 0: i2 = 0
    if k1 < 0: i3 = 0

    i2 = i + dpx
    j2 = j + dpy
    k2 = k + dpz
    if i2 > H: i2 = H
    if j2 > W: j2 = W
    if k2 > C: k2 = C

    for I in range(i1, i2):
        for J in range(j1, j2):
            for K in range(k1, k2):
                mask[H*W*K + I*W + J] = 0

    return 1

def updateList(mask, S, N, H, W, C):
    M = 0
    for c in range(N):
        idx = H*W*S[2][c] + S[0][c]*W + S[1][c]
        flg = mask[idx]
        if flg == 1:
            S[0][M] = S[0][c]
            S[1][M] = S[1][c]
            S[2][M] = S[2][c]
            M += 1

    return M

#----- kNN -----#

def kNN(L):
    #print("kNN")
    print("# of list points for mls:", L)

    return 0

#----- MLS -----#

def mls():
    #print("mls")

    return 0
