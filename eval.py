import numpy as np
import math
import random
import csv

def computeMAE(img1, img2, H, W, C):
    res = 0
    for k in range(C):
        for i in range(H):
            for j in range(W):
                res += abs( img2[k][i][j] - img1[k][i][j] )

    res = res / (H*W*C)

    return res


def read_landmark_info(file_name):
    res = np.zeros((3, 50), dtype=int)
    n   = 0

    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if n >= 1:
                res[0][n-1] = int( float(row[1]) )
                res[1][n-1] = int( float(row[2]) ) + 240
                res[2][n-1] = int( float(row[3]) )

            n = n + 1

    return n-1, res

def computeRobustness(moving, fixed, pred, file_name):
    n, arr = read_landmark_info(file_name)
    count = 0
    for c in range(n):
        k = arr[2][c]
        i = arr[0][c]
        j = arr[1][c]
        #print("indices:", i, j, k)
        k = 1 ##### for test #####
        tmp = abs(moving[k][i][j] - fixed[k][i][j]) - abs(pred[k][i][j] - fixed[k][i][j])
        if tmp < 0: count = count + 1

    r = count / n

    return r


def computeJacobiDeterminant(d, d_ws):

    return res
