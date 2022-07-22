import numpy as np
import math
import random
import csv

from ImgSeg import saveImg


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

def computeMAE(D, moving, fixed, H, W, C, file_name, csv_file_name):
    N, arr = read_landmark_info(file_name)
    AEs     = np.zeros(N)
    AEs_new = np.zeros(N)
    new_csv = np.zeros((N, 4))

    c = 0
    for n in range(N):
        i = arr[0][n]
        j = arr[1][n]
        k = arr[2][n]
        k = 1

        AEs[n] = abs( moving[k][i][j] - fixed[k][i][j] )

        ii = i - D[0][k*H*W + i*W +j]
        jj = j - D[1][k*H*W + i*W +j]
        kk = k - D[2][k*H*W + i*W +j]

        AEs_new[n] = abs( moving[kk][ii][jj] - fixed[k][i][j] )
        if AEs[n] > AEs_new[n]: c += 1

        #print(i, j, k, ii, jj, kk, AEs[n], AEs_new[n])
        new_csv[n][0] = n
        new_csv[n][1] = ii
        new_csv[n][2] = jj - 240
        new_csv[n][3] = kk

    MAE_before = np.median(AEs)
    MAE_after  = np.median(AEs_new)

    r = c / N

    print("MAE_before:", MAE_before, "MAE_after:", MAE_after, "Robustness:", r)

    # write to csv
    with open(csv_file_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("Landmark", "X", "Y", "Z"))
        writer.writerows(new_csv)

    return r

def computeJacobiDeterminant(d, d_ws, L, H, W, C, dpx, dpy, dpz, file_name, nii_file_name):
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

                A = 0
                F = tmp[2][idx]*tmp[0][idx]*tmp[1][idx]

                jd[c][h][w] = -1*A + F
                #print(jd[c][h][w])

                # collect #negative elements
                if jd[c][h][w] < 0: n += 1

    # - get statistics - #
    jd_max  = np.amax(jd)
    jd_mean = np.mean(jd)

    return n, jd, jd_max, jd_mean
