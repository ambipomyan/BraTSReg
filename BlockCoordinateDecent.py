import numpy as np
import math
import random

from utils import distance

# GPU settings for tests
BLOCKS  = 512
THREADS = 256
BUCKETS = 512

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
            flg = checkHit(mask_tmp, dpx, dpy, dpz, S_tmp[0][idx], S_tmp[1][idx], S_tmp[2][idx], H, W, C)
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

def checkHit(mask, dpx, dpy, dpz, i, j, k, H, W, C):
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

def kNN(src, L, S, N, KNN, xmm, ymm, zmm, knn):
    #print("kNN")
    print("# of list points for mls:", L)
    print("voxel dim by mm (HWC):", xmm, ymm, zmm)

    # init local KNN, Vals and buckets
    vals      = np.zeros(L*BLOCKS)
    localKNN  = np.zeros(knn*BLOCKS, dtype=int)
    localVals = np.zeros(knn*BLOCKS)
    buckets   = np.zeros(1024, dtype=int)

    # random pick loop indeces for tests only
    bid = random.randint(0, BLOCKS)
    tid = random.randint(0, THREADS)

    # compute knn
    for idx in range(0, N, BLOCKS):
        # for bid in range(BLOCKS): for tid in range(THREADS):
        computeDist(vals, idx, src, L, S, N, xmm, ymm, zmm, bid, tid)
        sortBucket(vals, L, localKNN, knn, xmm, ymm, zmm, bid, tid, buckets)
        formatDist(vals, localVals, localKNN, L, knn, bid, tid) # not necessarily needed

        count = 0
        for i in range(idx, idx+BLOCKS):
            if i < N:
                for k in range(knn):
                    KNN[i*knn + k] = localKNN[count]
                    count += 1
    
    return 0

def computeDist(vals, idx, src, L, S, N, xmm, ymm, zmm, bid, tid):
    pid = bid + idx

    if pid < N:
        for i in range(tid, L, THREADS):
           di = xmm*(src[0][i] - S[0][pid])
           dj = ymm*(src[1][i] - S[1][pid])
           dk = zmm*(src[2][i] - S[2][pid])
           dist = math.sqrt(di*di + dj*dj + dk*dk)

           vals[bid*L + i] = dist

    return 0

def sortBucket(vals, L, KNN, knn, xmm, ymm, zmm, bid, tid, buckets):
    buckets[tid] = 0
    buckets[tid + 512] = 0

    # first pass
    for i in range(tid, L, BUCKETS):
        val = vals[bid*L + i]
        d = int(round(val / xmm)) # suppose that xmm == ymm
        if d >= BUCKETS:
            buckets[BUCKETS] += 1
        else: 
            buckets[d + 1] += 1

    p_in  = 0
    p_out = 1
    for i in range(9):
        offset = pow(2, i)
        p_out = 1 - p_out # swap p_in and p_out
        p_in  = 1 - p_out

        if tid >= offset:
            buckets[512*p_out + tid] = buckets[512*p_in + tid] + buckets[512*p_in + tid - offset]
        else:
            buckets[512*p_out + tid] = buckets[512*p_in + tid]

    buckets[tid] = buckets[512*p_out + tid]

    # second pass
    for i in range(tid, L, BUCKETS):
        val = vals[bid*L + i]
        d = int(round(val / xmm)) # suppose that xmm == ymm
        if d >= BUCKETS:
            buckets[BUCKETS - 1] += 1
        else: 
            buckets[d] += 1

        count = buckets[d]
        if buckets[d] < knn: buckets[count + 512] = i

    if tid < knn: KNN[bid*knn + tid] = buckets[tid + 512]

    return 0

def formatDist(vals, localVals, KNN, L, knn, bid, tid):
    k = KNN[bid*knn + tid]
    localVals[bid*knn + tid] = vals[bid*L + k]

    return 0


#----- MLS -----#

def mls():
    #print("mls")

    return 0
