import numpy as np
import math
import random

# ----- computeFunctinonRes ----- #

def computeFuncRes(A, KNN, knn, b, x, r, p, Ap, Z, Y, L, alpha, mu):
    #print("compute objective function...")
    for i in range(3):
        for j in range(L): b[j] = Z[i][j]/mu
        cg(A, KNN, knn, b, x, r, p, Ap, L, alpha, mu)
        for j in range(L): Y[i][j] = x[j]

    return 0

def cg(A, KNN, knn, b, x, r, p, Ap, L, alpha, mu):

    return 0

# ----- updateDisplacementField ----- #

def updateDisplacementField(F, localVals, z_ws, Z, Y, L, alpha, mu, sx, sy, sz, rx, ry, rz):
    #print("update displacement field...")
    obj = 0
    cc  = 0

    return obj, cc

# ----- computeIterDiff ----- #

def computeIterDiff(Z, Zold, Y, L):
    #print("compute difference between itrerations...")
    nrmABS = 0
    nrmZ  = 0

    for i in range(3):
        for j in range(L):
            # MAE
            nrmABS += abs(Z[i][j] - Y[i][j]) / L
            # diff between iters
            nrmZ   += abs(Z[i][j] - Zold[i][j]) / L
            # update Zold
            Zold[i][j] = Z[i][j]

    return nrmZ, nrmABS
