import numpy as np
import math
import random

from utils import multMat, multVec, axpby

# block matching settings for tests
BLOCKS  = 512
THREADS = 256
BUCKETS = 512

# rx = ry = 3, rz = 1; 147 = 7 x 7 x 3
MEMSIZE = 147

# GPU parallelism
from numba import cuda, int32, float32


# ----- computeFunctinonRes ----- #

def computeFuncRes(A, KNN, knn, b, x, r, p, Ap, Z, Y, L, alpha, mu):
    #print("update objective function...")
    for i in range(3):
        for j in range(L): b[j] = mu*Z[i][j]
        cg(A, KNN, knn, b, x, r, p, Ap, L, alpha, mu)
        for j in range(L): Y[i][j] = x[j]

    return 0

def cg(A, KNN, knn, b, x, r, p, Ap, L, alpha, mu):
    rTr = initCG(x, r, b, p, L)
    #print("rTr:", rTr)
    tol = 0.005
    if rTr < tol: return 0

    max_iter = L
    for i in range(max_iter):
        # update x
        multMat(A, KNN, knn, p, Ap, b, L, mu)
        a = rTr/multVec(Ap, p, L)
        axpby(x, 1, x, a, p, L)

        # update r
        axpby(r, 1, r, a, Ap, L)
        rTr_new = multVec(r, r, L)
        #print("rTr_new:", rTr_new)

        # check convergency
        if rTr_new < tol: break

        # update p
        beta = rTr_new/rTr
        rTr  = rTr_new
        axpby(p, -1, r, beta, p, L)

    return 0

def initCG(x, r, b, p, L):
    vals = np.zeros(THREADS)
    for tid in range(THREADS):
        vals[tid] = 0
        for i in range(tid, L, THREADS):
            r[i] = -1*b[i]
            p[i] = b[i]
            x[i] = 0

            vals[tid] += b[i]**2

    rTr = 0
    for tid in range(THREADS):
        rTr += vals[tid]

    return rTr

# ----- updateDisplacementField ----- #

def updateDisplacementField(fixed, moving, F, I, S, Z, Y, L, localVals, mu, sx, sy, sz, rx, ry, rz, H, W, C):
    #print("update displacement field...")
    obj = 0
    cc  = 0

    count = 0
    while count < L:
        # search for block matching
        #searchMin[BLOCKS, THREADS](fixed, moving, count, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz, H, W, C)
        searchMin_serial(fixed, moving, count, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz, H, W, C)

        # sort for minimizers
        sortMin[BLOCKS, THREADS](count, F, I, Z, L, localVals, sx, sy, sz)

        for i in range(BLOCKS):
            obj += localVals[0][i]
            cc  += localVals[1][i]

        count += BLOCKS

    for i in range(3):
        for j in range(L):
            Z[i][j] += int( round(Y[i][j]) )

    return obj, cc

# CUDA kernels

@cuda.jit
def searchMin(fixed, moving, idx, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz, H, W, C):
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    pid = bid + idx

    SN = (2*sx + 1)*(2*sy + 1)*(2*sz + 1)
    RN = (2*rx + 1)*(2*ry + 1)*(2*rz + 1)
    #MEMSIZE = RN

    # shared memory
    vals = cuda.shared.array(shape=(MEMSIZE), dtype=int32)

    if pid < L:
        #print("S[0], S[1], S[2], pid:", S[0][pid], S[1][pid], S[2][pid], pid)
        src0 = S[0][pid]
        src1 = S[1][pid]
        src2 = S[2][pid]
        #print("src0, src1, src2:", src0, src1, src2)

        #print("Y[0], Y[1], Y[2], pid:", Y[0][pid], Y[1][pid], Y[2][pid], pid)
        d0 = Y[0][pid]
        d1 = Y[1][pid]
        d2 = Y[2][pid]
        #print("d0, d1, d2:", d0, d1, d2)

        tar0 = src0 + int( round(d0) )
        tar1 = src1 + int( round(d1) )
        tar2 = src2 + int( round(d2) )
        #print("tar0, tar1, tar2:", tar0, tar1, tar2)

        d0 += src0 - tar0
        d1 += src1 - tar1
        d2 += src2 - tar2
        #print("d0, d1, d2:", d0, d1, d2)

        p_count = 0
        if tid == 0:
            for k in range(-rz, rz+1):
                for i in range(-rx, rx+1):
                    for j in range(-ry, ry+1):
                        # get intensity vals
                        t_x = i + src0
                        t_y = j + src1
                        t_z = k + src2
                        if t_x >= H: t_x = H - 1
                        if t_y >= W: t_y = W - 1
                        if t_z >= C: t_z = C - 1
                        vals[p_count] = moving[t_z][t_x][t_y]
                        p_count += 1
        # _syncthreads, take care for the indent style!
        cuda.syncthreads()

        minVal0 = 100000000
        for count in range(tid, SN, THREADS):
            # based on the assumption: sx == sy
            ti = int( (count%(2*sx + 1)**2)/(2*sx + 1) - sx )
            tj = int( (count%(2*sx + 1)**2)%(2*sx + 1) - sx )
            tk = int(  count/(2*sx + 1)**2             - sz )
            #print("tk, ti, tj:", tk, ti, tj)

            nrm = (ti - d0)**2 + (tj - d1)**2 + (tk - d2)**2

            ti += tar0
            tj += tar1
            tk += tar2
            #print("tk+tar, ti+tar, tj+tar:", tk, ti, tj)

            x  = 0
            y  = 0
            x2 = 0
            y2 = 0
            xy = 0

            p_count = 0
            for k in range(-rz, rz+1):
                for i in range(-rx, rx+1):
                    for j in range(-ry, ry+1):
                        p = vals[p_count]
                        t_x = i + ti
                        t_y = j + tj
                        t_z = k + tk
                        if t_x >= H: t_x = H - 1
                        if t_y >= W: t_y = W - 1
                        if t_z >= C: t_z = C - 1
                        q = fixed[t_z][t_x][t_y]

                        x  += p
                        y  += q
                        x2 += p*p
                        y2 += q*q
                        xy += p*q

                        p_count += 1

            # objective function value
            if (x2 - x**2/RN) <= 0 or (y2 - y**2/RN) <= 0:
                #print("NaN or inf encountered: x2:", x2, "x:", x, "y2:", y2, "y:", y, "RN:", RN)
                val1 = 1
            else:
                val1 = (xy - x*y/RN) / ( math.sqrt(x2 - x**2/RN)*math.sqrt(y2 - y**2/RN) )
                val1 = 1 - val1**2

            val0 = val1 + mu*nrm

            # update minVal
            if minVal0 > val0:
                minVal0 = val0
                minVal1 = val1
                idx_tmp = count
        # end of count for loop

        F[0][bid*THREADS + tid] = minVal0
        F[1][bid*THREADS + tid] = minVal1
        I[bid*THREADS + tid]    = idx_tmp
    # end of if pid < L

    # no returns 
    #return 0


@cuda.jit
def sortMin(idx, F, I, Z, L, localVals, sx, sy, sz):
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    pid = bid + idx

    if tid == 0:
        localVals[0][bid] = 0
        localVals[1][bid] = 0

    vals    = cuda.shared.array(shape=(2, THREADS), dtype=float32)
    idx_tmp = cuda.shared.array(shape=(THREADS),    dtype=int32)

    if pid < L:
        vals[0][tid] = F[0][bid*THREADS + tid]
        vals[1][tid] = F[1][bid*THREADS + tid]
        idx_tmp[tid] = I[bid*THREADS + tid]
        ID = int(round(THREADS/2))

        cuda.syncthreads()

        while ID != 0:
            if tid < ID:
                if vals[0][tid] > vals[0][tid + ID]:
                    vals[0][tid] = vals[0][tid + ID]
                    vals[1][tid] = vals[1][tid + ID]
                    idx_tmp[tid] = idx_tmp[tid + ID]

            cuda.syncthreads()

            ID = int(round(ID/2))

        # Update solution of the displacement field
        if tid == 0:
            ID = idx_tmp[0]
            localVals[0][bid] = vals[0][0]
            localVals[1][bid] = vals[1][0]

            # based on the assumption: sx == sy
            sol0 = int( (ID%(2*sx + 1)**2)/(2*sx + 1) - sx )
            sol1 = int( (ID%(2*sx + 1)**2)%(2*sx + 1) - sx )
            sol2 = int(  ID/(2*sx + 1)**2             - sz )

            Z[0][pid] = sol0
            Z[1][pid] = sol1
            Z[2][pid] = sol2



# ----- computeIterDiff ----- #

def computeIterDiff(Z, Zold, Y, L):
    #print("compute difference between itrerations...")
    nrmABS = 0
    nrmZ   = 0

    for i in range(3):
        for j in range(L):
            # MAE
            nrmABS += abs(Z[i][j] - Y[i][j])
            # diff between iters
            nrmZ   += abs(Z[i][j] - Zold[i][j])
            # update Zold
            Zold[i][j] = Z[i][j]

    return nrmZ, nrmABS

# searchMin_serial
def searchMin_serial(fixed, moving, idx, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz, H, W, C):
    SN = (2*sx + 1)*(2*sy + 1)*(2*sz + 1)
    RN = (2*rx + 1)*(2*ry + 1)*(2*rz + 1)
    MEMSIZE = RN

    for bid in range(BLOCKS):
        src    = np.zeros(3, dtype=int)
        tar    = np.zeros(3, dtype=int)
        d      = np.zeros(3)
        minVal = np.zeros(2)
        val    = np.zeros(2)
        vals   = np.zeros(MEMSIZE, dtype=int)

        pid = bid + idx
        if pid < L:
            #print("S[0], S[1], S[2], pid:", S[0][pid], S[1][pid], S[2][pid], pid)

            src[0] = S[0][pid]
            src[1] = S[1][pid]
            src[2] = S[2][pid]

            #print("src[0], src[1], src[2]:", src[0], src[1], src[2])

            d[0] = Y[0][pid]
            d[1] = Y[1][pid]
            d[2] = Y[2][pid]

            #print("Y[0], Y[1], Y[2], pid:", Y[0][pid], Y[1][pid], Y[2][pid], pid)

            tar[0] = src[0] + int( round(d[0]) )
            tar[1] = src[1] + int( round(d[1]) )
            tar[2] = src[2] + int( round(d[2]) )

            #print("tar[0], tar[1], tar[2]:", tar[0], tar[1], tar[2])

            d[0] += src[0] - tar[0]
            d[1] += src[1] - tar[1]
            d[2] += src[2] - tar[2]

            #print("d[0], d[1], d[2]:", d[0], d[1], d[2])

            p_count = 0
            for k in range(-rz, rz+1):
                for i in range(-rx, rx+1):
                    for j in range(-ry, ry+1):
                        # get intensity vals
                        t_x = i + src[0]
                        t_y = j + src[1]
                        t_z = k + src[2]
                        if t_x >= H: t_x = H - 1
                        if t_y >= W: t_y = W - 1
                        if t_z >= C: t_z = C - 1
                        vals[p_count] = moving[t_z][t_x][t_y]
                        p_count += 1

            minVal[0] = 1000000
            for tid in range(THREADS):
                for count in range(tid, SN, THREADS):
                    # based on the assumption: sx == sy
                    ti = int( (count%(2*sx + 1)**2)/(2*sx + 1) - sx )
                    tj = int( (count%(2*sx + 1)**2)%(2*sx + 1) - sx )
                    tk = int(  count/(2*sx + 1)**2             - sz )

                    #print("tk, ti, tj:", tk, ti, tj)

                    nrm = (ti - d[0])**2 + (tj - d[1])**2 + (tk - d[2])**2

                    ti += tar[0]
                    tj += tar[1]
                    tk += tar[2]

                    #print("tk+tar, ti+tar, tj+tar:", tk, ti, tj)

                    x  = 0
                    y  = 0
                    x2 = 0
                    y2 = 0
                    xy = 0

                    p_count = 0
                    for k in range(-rz, rz+1):
                        for i in range(-rx, rx+1):
                            for j in range(-ry, ry+1):
                                p = vals[p_count]
                                t_x = i + ti
                                t_y = j + tj
                                t_z = k + tk
                                if t_x >= H: t_x = H - 1
                                if t_y >= W: t_y = W - 1
                                if t_z >= C: t_z = C - 1
                                q = fixed[t_z][t_x][t_y]

                                x  += p
                                y  += q
                                x2 += p*p
                                y2 += q*q
                                xy += p*q

                                p_count += 1

                    # objective function value
                    if (x2 - x**2/RN) <= 0 or (y2 - y**2/RN) <= 0:
                        #print("NaN or inf encountered: x2:", x2, "x:", x, "y2:", y2, "y:", y, "RN:", RN)
                        val[1] = 1
                    else:
                        tmp    = (xy - x*y/RN) / ( math.sqrt(x2 - x**2/RN)*math.sqrt(y2 - y**2/RN) )
                        val[1] = 1 - tmp**2
                        
                    val[0] = val[1] + mu*nrm
                    
                    # update minVal
                    if minVal[0] > val[0]:
                        minVal  = val
                        idx_tmp = count

                F[0][bid*THREADS + tid] = minVal[0]
                F[1][bid*THREADS + tid] = minVal[1]

                I[bid*THREADS + tid]    = idx_tmp
            # end of tid loop
        # end of if pid < L
    # end of bid loop

    return 0
