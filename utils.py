import numpy as np
import math
import random

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

def CholeskyFactorization(b, R, dim):
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
    v1 = np.zeros(dim)
    for i in range(dim):
        for j in range(i):
            for k in range(dim):
                v1[k] += R[i][j]*b[j][k]
        for s in range(dim):
            b[i][s] = (b[i][s] - v1[s])/R[i][i]

    # backward
    v2 = np.zeros(dim)
    for i in range(dim-1, -1, -1):
        for j in range(i+1, dim):
            for k in range(dim):
                v2[k] += R[i][j]*b[j][k]
        for s in range(dim):
            b[i][s] = (b[i][s] - v2[s])/R[i][i]

    return 0
