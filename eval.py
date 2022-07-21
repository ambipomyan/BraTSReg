import numpy as np
import math
import random
import csv

from ImgSeg import saveImg

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
        #
        k = 1
        #
        tmp = abs(moving[k][i][j] - fixed[k][i][j]) - abs(pred[k][i][j] - fixed[k][i][j])
        if tmp < 0: count = count + 1

    r = count

    return r, n


def computeJacobiDeterminant(d, d_ws, L, H, W, C, dpx, dpy, dpz, file_name):
    n = 0
    jd = np.zeros((C, H, W))
    tmp = np.zeros((3, H*W*C), dtype=int)

    # - recover displacement field - #
    for l in range(L):
        idx_z = d_ws[2][l]
        idx_x = d_ws[0][l]
        idx_y = d_ws[1][l]

        for k in range(-dpz, dpz+1):
            for i in range(-dpx, dpx+1):
                for j in range(-dpy, dpy+1):
                    c = idx_z + k
                    h = idx_x + i
                    w = idx_y + j

                    if c >= C: c = C - 1
                    if h >= H: h = H - 1
                    if w >= W: w = W - 1

                    if c < 0: c = 0
                    if h < 0: h = 0
                    if w < 0: w = 0

                    ID = c*H*W + W*h + w

                    tmp[2][ID] = d[2][l]
                    tmp[0][ID] = d[0][l]
                    tmp[1][ID] = d[1][l]

    # - compute JD - #
    for c in range(C):
        for h in range(H):
            for w in range(W):
                idx = c*H*W + W*h + w

                # compute JD elements

                #          | a11 a12 a13 |
                # det(J) = | a21 a22 a23 |
                #          | a31 a32 a33 |

                # a11 = d(tx)/d(x)
                # a13 = d(tx)/d(z)
                # a33 = d(tz)/d(z)

                A = -c*h*w
                F = (c + tmp[2][idx])*(h + tmp[0][idx])*(w + tmp[1][idx])

                jd[c][h][w] = A + F
                #print(jd[c][h][w])

                # collect #negative elements
                if jd[c][h][w] < 0: n += 1

    # - get statistics - #
    jd_max  = np.amax(jd)
    jd_mean = np.mean(jd)

    return n, jd, jd_max, jd_mean
