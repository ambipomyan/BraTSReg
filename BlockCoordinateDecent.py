import numpy as np
import math

# block matching settings for tests
BLOCKS  = 512
THREADS = 256
BUCKETS = 512

from utils import randomPickInt, initUpperTriangleMatrix, initIdMatrix, CholeskyFactorization, compute2Norm

# GPU parallelism
from numba import cuda, int32, float32


#----- dart throw -----#

def throwDarts(mask, S, dpx, dpy, dpz, H, W, C, Kid):
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

    print("throwing darts...")
    count = 0
    M = N
    numDarts = 10000
    while M != 0:
        for n in range(numDarts):
            idx = randomPickInt(M)
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
    for k in range(C):
        for i in range(H):
            for j in range(W):
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
    if i2 >= H: i2 = H - 1
    if j2 >= W: j2 = W - 1
    if k2 >= C: k2 = C - 1

    for K in range(k1, k2+1):
        for I in range(i1, i2+1):
            for J in range(j1, j2+1):
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

def kNN(src, L, S, N, KNN, knn, xmm, ymm, zmm):
    #print("kNN")
    print("# of list points for mls:", L)
    print("voxel dim by mm (HWC):", xmm, ymm, zmm)

    # init local KNN, Vals and buckets
    vals      = np.zeros(L*BLOCKS)
    localKNN  = np.zeros(knn*BLOCKS, dtype=int)
    localVals = np.zeros(knn*BLOCKS)

    # compute knn
    for idx in range(0, N, BLOCKS):
        computeDist[BLOCKS, THREADS](vals, idx, src, L, S, N, xmm, ymm, zmm)
        countBuckets[BLOCKS, BUCKETS](vals, L, localKNN, knn, xmm, ymm, zmm)
        #formatDist[BLOCKS, knn](vals, localVals, localKNN, L, knn) # not necessarily needed

        count = 0
        for i in range(idx, idx+BLOCKS):
            if i < N:
                for k in range(knn):
                    KNN[i*knn + k] = localKNN[count]
                    count += 1

    return 0

# CUDA kernels

@cuda.jit
def computeDist(vals, idx, src, L, S, N, xmm, ymm, zmm):
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    pid = bid + idx

    if pid < N:
        for i in range(tid, L, THREADS):
            di = xmm*(src[0][i] - S[0][pid])
            dj = ymm*(src[1][i] - S[1][pid])
            dk = zmm*(src[2][i] - S[2][pid])

            dist = math.sqrt(di**2 + dj**2 + dk**2)

            vals[bid*L + i] = dist

    #return 0

@cuda.jit
def countBuckets(vals, L, KNN, knn, xmm, ymm, zmm):
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # shared memory
    buckets = cuda.shared.array(shape=(1024), dtype=int32) # int32 is used for atomic add
    buckets[tid] = 0
    buckets[tid + 512] = 0

    cuda.syncthreads()

    # first pass
    for i in range(tid, L, BUCKETS):
        val = vals[bid*L + i]
        # suppose that xmm == ymm
        d = int(round(val / xmm))
        if d >= BUCKETS: d = BUCKETS - 1

        # update bucket, atomic add
        cuda.atomic.add(buckets, d + 1, 1)
        #print("1st pass - d:", d, "d+1:", d+1, "buckets[d+1]:", buckets[d + 1], "i:", i)

    cuda.syncthreads()

    p_in  = 0
    p_out = 1
    for i in range(9): # 9 = log_2(512)
        offset = 2**i
        #print("offset:", offset)
        p_out = 1 - p_out # swap p_in and p_out
        p_in  = 1 - p_out

        if tid >= offset:
            buckets[512*p_out + tid] = buckets[512*p_in + tid] + buckets[512*p_in + tid - offset]
        else:
            buckets[512*p_out + tid] = buckets[512*p_in + tid]

        cuda.syncthreads()

    buckets[tid] = buckets[512*p_out + tid]
    #print("buckets[tid]:", buckets[tid], "tid:", tid)

    cuda.syncthreads()


    # second pass
    for i in range(tid, L, BUCKETS):
        val = vals[bid*L + i]
        # suppose that xmm == ymm
        d = int(round(val / xmm))

        if d >= BUCKETS: d = BUCKETS - 1
        # caution: atomic add function returns the OLD val!
        count = cuda.atomic.add(buckets, d, 1)
        #print("2nd pass - d:", d, "buckets[d]:", count, "i:", i)

        if count < knn: buckets[count + 512] = i

    cuda.syncthreads()


    # update KNN
    if tid < knn: KNN[bid*knn + tid] = buckets[tid + 512]
    #print("KNN[bid*knn + tid]:", KNN[bid*knn + tid], "bid*knn + tid:", bid*knn + tid)


@cuda.jit
def formatDist(vals, localVals, KNN, L, knn):
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    k = KNN[bid*knn + tid]
    localVals[bid*knn + tid] = vals[bid*L + k]



#----- MLS -----#

def mls(S, L, KNN, A, knn, xmm, ymm,zmm):
    #print("mls")
    nrm  = np.zeros(knn) # 2-norm
    ws   = np.zeros(4)
    ATA  = np.zeros(10)  # 4x4 matrix (upper part)
    xTAI = np.zeros(4)   # 4x1 array

    for count in range(L):
        # w
        for k in range(knn):
            idx = KNN[count*knn + k]
            nrm[k] = compute2Norm(S, idx, count, xmm, ymm, zmm)
        #if count == 0: print(nrm)

        h  = np.max(nrm) # infinity-norm
        h2 = h**2
        #if count == 0: print("h:", h)

        for k in range(knn):
            nrm[k] = math.exp(-nrm[k]*1/h2)

        # A'A
        for k in range(knn):
            idx = KNN[count*knn + k]
            ws[0] = nrm[k]*(S[0][idx] - S[0][count]) # x
            ws[1] = nrm[k]*(S[1][idx] - S[1][count]) # y
            ws[2] = nrm[k]*(S[2][idx] - S[2][count]) # z
            ws[3] = nrm[k]                           # w

            ATA[0] += ws[0]*ws[0]
            ATA[1] += ws[0]*ws[1]
            ATA[2] += ws[0]*ws[2]
            ATA[3] += ws[0]*ws[3]
            ATA[4] += ws[1]*ws[1]
            ATA[5] += ws[1]*ws[2]
            ATA[6] += ws[1]*ws[3]
            ATA[7] += ws[2]*ws[2]
            ATA[8] += ws[2]*ws[3]
            ATA[9] += ws[3]*ws[3]

        #print(ATA)
        CholeskyFactorization(xTAI, ATA, 4)

        # x'A'AIA'w
        for k in range(knn):
            idx = KNN[count*knn + k]
            ws[0] = nrm[k]*(S[0][idx] - S[0][count]) # x
            ws[1] = nrm[k]*(S[1][idx] - S[1][count]) # y
            ws[2] = nrm[k]*(S[2][idx] - S[2][count]) # z
            ws[3] = nrm[k]                           # w

            A[count*knn + k] = nrm[k]*( xTAI[0]*ws[0] + \
                                        xTAI[1]*ws[1] + \
                                        xTAI[2]*ws[2] + \
                                        xTAI[3]*ws[3]    )

    return 0
