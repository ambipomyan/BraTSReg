import numpy as np
import math
import random

# block matching settings for tests
BLOCKS  = 512
THREADS = 512
BUCKETS = 512

def randomPickInt(N):
    res = random.randint(0, N-1)

    return res

def initUpperTriangleMatrix(arr, dim):
    M = np.zeros((dim, dim))
    M[0][0] = arr[0]
    M[0][1] = arr[1]
    M[0][2] = arr[2]
    M[0][3] = arr[3]
    M[1][1] = arr[4]
    M[1][2] = arr[5]
    M[1][3] = arr[6]
    M[2][2] = arr[7]
    M[2][3] = arr[8]
    M[3][3] = arr[9]
    M[1][0] = M[0][1]
    M[2][0] = M[0][2]
    M[2][1] = M[1][2]
    M[3][0] = M[0][3]
    M[3][1] = M[1][3]
    M[3][2] = M[2][3]

    return M

def initIdMatrix(dim):
    Diag = np.eye(dim)

    return Diag

def CholeskyFactorization(xTAI, ATA, dim):
    R = initUpperTriangleMatrix(ATA, dim)
    b = initIdMatrix(dim)    

    R[0][0] = math.sqrt(R[0][0])

    R[1][0] = R[1][0]/R[0][0]
    R[0][1] = R[1][0]
    R[1][1] = math.sqrt(R[1][1] - R[1][0]**2)

    R[2][0] = R[2][0]/R[0][0]
    R[0][2] = R[2][0]
    R[2][1] = (R[2][1] - R[2][0]*R[1][0])/R[1][1]
    R[1][2] = R[2][1]
    R[2][2] = math.sqrt(R[2][2] - R[2][1]**2 - R[2][0]**2)

    R[3][0] = R[3][0]/R[0][0]
    R[0][3] = R[3][0]
    R[3][1] = (R[3][1] - R[3][0]*R[1][0])/R[1][1]
    R[1][3] = R[3][1]
    R[3][2] = (R[3][2] - R[3][0]*R[2][0] - R[3][1]*R[2][1])/R[2][2]
    R[2][3] = R[3][2]
    R[3][3] = math.sqrt(R[3][3] - R[3][2]**2 - R[3][1]**2 - R[3][0]**2)

    # forward
    for i in range(dim):
        v = np.zeros(dim)

        for j in range(i):
            for k in range(dim):
                v[k] += R[i][j]*b[j][k]

        for s in range(dim):
            b[i][s] = (b[i][s] - v[s])/R[i][i]

    # backward
    for i in range(dim-1, -1, -1):
        v = np.zeros(dim)

        for j in range(i+1, dim):
            for k in range(dim):
                v[k] += R[i][j]*b[j][k]

        for s in range(dim):
            b[i][s] = (b[i][s] - v[s])/R[i][i]

    xTAI[0] = b[3][0]
    xTAI[1] = b[3][1]
    xTAI[2] = b[3][2]
    xTAI[3] = b[3][3]

    return 0

def compute2Norm(S, i, j, xmm, ymm, zmm):
    res = math.sqrt( ((S[0][i] - S[0][j])*xmm)**2 + \
                     ((S[1][i] - S[1][j])*ymm)**2 + \
                     ((S[2][i] - S[2][j])*zmm)**2    )

    return res

def multMat(A, KNN, knn, x, y, z, L, mu):
    multA(A, KNN, knn, x, z, L)
    multAT(A, KNN, knn, z, y, L)

    axpby(y, 1, y, mu, x, L)

    return 0

def multA(A, KNN, knn, x, z, L):
    for tid in range(THREADS):
        for i in range(tid, L, THREADS):
            val = 0
            for j in range(knn):
                idx  = KNN[i*knn + j]
                val += A[i*knn + j]*x[idx]

            z[i] = val - x[i]

    return 0

def multAT(A, KNN, knn, z, y, L):
    for i in range(L):
        y[i] = -z[i]

    for tid in range(THREADS):
        for i in range(tid, L, THREADS):
            for j in range(knn):
                idx     = KNN[i*knn + j]
                y[idx] += A[i*knn + j]*z[i]

    return 0

def multVec(x, y, L):
    # inner product
    vals = np.zeros(THREADS)
    for tid in range(THREADS):
        vals[tid] = 0
        for i in range(tid, L, THREADS):
            vals[tid] += x[i]*y[i]

    val = 0
    for tid in range(THREADS):
        val += vals[tid]

    return val

def axpby(z, a, x, b, y, L):
    # z = ax + by
    for tid in range(THREADS):
        for i in range(tid, L, THREADS):
            z[i] = a*x[i] + b*y[i]

    return 0
