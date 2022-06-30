import numpy as np
import math
import random

from utils import multMat, multVec, axpby

# block matching settings for tests
BLOCKS  = 512
THREADS = 256
BUCKETS = 512


# ----- computeFunctinonRes ----- #

def computeFuncRes(A, KNN, knn, b, x, r, p, Ap, Z, Y, L, alpha, mu):
    #print("update objective function...")
    for i in range(3):
        for j in range(L): b[j] = Z[i][j]/mu
        cg(A, KNN, knn, b, x, r, p, Ap, L, alpha, mu)
        for j in range(L): Y[i][j] = x[j]

    return 0

def cg(A, KNN, knn, b, x, r, p, Ap, L, alpha, mu):
    rTr = initCG(x, r, b, p, L)
    print("rTr:", rTr)
    tol = 0.001
    if rTr < tol: return 0

    max_iter = L
    for i in range(max_iter):
        # update x
        multMat(A, KNN, knn, b, p, Ap, L, mu)
        a = rTr/multVec(Ap, p, L)
        axpby(x, 1, x, a, p, L)

        # update r
        axpby(r, 1, r, a, Ap, L)
        rTr_new = multVec(r, r, L)
        print("rTr_new:", rTr_new)

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

def updateDisplacementField(fixed, moving, F, I, S, Z, Y, L, localVals, mu, sx, sy, sz, rx, ry, rz):
    #print("update displacement field...")
    obj = 0
    cc  = 0

    count = 0
    while count < L:
        # search for block matching
        searchMin(fixed, moving, count, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz)

        # sort for minimizers
        sortMin(count, F, I, Z, L, localVals, sx, sy, sz)

        for i in range(BLOCKS):
            obj += localVals[0][i]
            cc  += localVals[1][i]

        count += BLOCKS

    for i in range(3):
        for j in range(L):
            Z[i][j] += round(Y[i][j])

    return obj, cc

def searchMin(fixed, moving, idx, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz):
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
            print("S[0], S[1], S[2], pid:", S[0][pid], S[1][pid], S[2][pid], pid)

            src[0] = S[0][pid]
            src[1] = S[1][pid]
            src[2] = S[2][pid]

            print("src[0], src[1], src[2]:", src[0], src[1], src[2])

            d[0] = Y[0][pid]
            d[1] = Y[1][pid]
            d[2] = Y[2][pid]

            print("Y[0], Y[1], Y[2], pid:", Y[0][pid], Y[1][pid], Y[2][pid], pid)

            tar[0] = src[0] + round(d[0])
            tar[1] = src[1] + round(d[1])
            tar[2] = src[2] + round(d[2])

            print("tar[0], tar[1], tar[2]:", tar[0], tar[1], tar[2])

            d[0] += src[0] - tar[0]
            d[1] += src[1] - tar[1]
            d[2] += src[2] - tar[2]

            print("d[0], d[1], d[2]:", d[0], d[1], d[2])

            p_count = 0
            for k in range(-rz, rz):
                for i in range(-rx, rx):
                    for j in range(-ry, ry):
                        # get intensity vals
                        vals[p_count] = moving[k + src[2]][i + src[0]][j + src[1]]
                        p_count += 1

            minVal[0] = 1000000
            for tid in range(THREADS):
                for count in range(tid, SN, THREADS):
                    # based on the assumption: sx == sy
                    ti = int( (count%(2*sx + 1)**2)/(2*sx + 1) - sx )
                    tj = int( (count%(2*sx + 1)**2)%(2*sx + 1) - sx )
                    tk = int(  count/(2*sx + 1)**2 - sz )

                    nrm = (ti - d[0])**2 + (tj - d[1])**2 + (tk - d[2])**2

                    ti += tar[0]
                    tj += tar[1]
                    tk += tar[2]
                    print("tk, ti, tj:", tk, ti, tj)

                    x  = 0
                    y  = 0
                    x2 = 0
                    y2 = 0
                    xy = 0

                    p_count = 0
                    for k in range(-rz, rz):
                        for i in range(-rx, rx):
                            for j in range(-ry, ry):
                                p = vals[p_count]
                                q = fixed[k + tk][i + ti][j + tj]

                                x  += p
                                y  += q
                                x2 += p*p
                                y2 += q*q
                                xy += p*q

                                p_count += 1

                    # objective function value
                    if (x2 - x**2/RN) <= 0 or (y2 - y**2/RN) <= 0:
                        #print("NaN or inf encountered: x2:", x2, "x:", x, "y2:", y2, "y:", y, "RN:", RN)
                        val[0] = 1 + 1/(2*mu)*nrm
                        val[1] = 1
                    else:
                        tmp    = (xy - x*y/RN) / ( math.sqrt(x2 - x**2/RN)*math.sqrt(y2 - y**2/RN) )
                        val[0] = tmp + 1/(2*mu)*nrm
                        val[1] = 1 - tmp**2

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

def sortMin(idx, F, I, Z, L, localVals, sx, sy, sz):
    for bid in range(BLOCKS):
        sol = np.zeros(3, dtype=int)

        localVals[0][bid] = 0
        localVals[1][bid] = 0

        vals    = np.zeros((2, THREADS))
        idx_tmp = np.zeros(THREADS, dtype=int)

        pid = bid + idx
        if pid < L:
            for tid in range(THREADS):
                vals[0][tid] = F[0][bid*THREADS + tid]
                vals[1][tid] = F[1][bid*THREADS + tid]
                idx_tmp[tid] = I[bid*THREADS + tid]

            ID = int(round(THREADS/2))
            while ID != 0:
                for tid in range(THREADS):
                    if tid < ID:
                        if vals[0][tid] > vals[0][tid + ID]:
                           vals[0][tid] = vals[0][tid + ID]
                           vals[1][tid] = vals[1][tid + ID]
                           idx_tmp[tid] = idx_tmp[tid + ID]

                ID = int(round(ID/2))

            # Update solution of the displacement field
            ID = idx_tmp[0]
            localVals[0][bid] = vals[0][0]
            localVals[1][bid] = vals[1][0]

            sol[0] = (ID%(2*sx + 1)**2)/(2*sx + 1) - sx
            sol[1] = (ID%(2*sy + 1)**2)%(2*sy + 1) - sy
            sol[2] =  ID/(2*sz + 1)**2 - sz

            Z[0][pid] = sol[0]
            Z[1][pid] = sol[1]
            Z[2][pid] = sol[2]

    return 0


# ----- computeIterDiff ----- #

def computeIterDiff(Z, Zold, Y, L):
    #print("compute difference between itrerations...")
    nrmABS = 0
    nrmZ   = 0

    for i in range(3):
        for j in range(L):
            # MAE
            nrmABS += abs(Z[i][j] - Y[i][j]) / L
            # diff between iters
            nrmZ   += abs(Z[i][j] - Zold[i][j]) / L
            # update Zold
            Zold[i][j] = Z[i][j]

    return nrmZ, nrmABS
