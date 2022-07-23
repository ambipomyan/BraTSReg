import numpy as np
import math
import random
import csv

import nibabel as nib

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

def save_JD_info(jd, H, W, C, file_name):
    print("saving nii.gz file...")
    # create nifti image
    nii_img = nib.Nifti1Image(jd, np.eye(4))
    # save image to file
    nib.save(nii_img, file_name)

    return 0

# ----- evaluation! ----- #

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

        #
        #k = 1 # ---- test only ----- #
        #

        AEs[n] = abs( moving[k][i][j] - fixed[k][i][j] )

        ii = i - D[0][k*H*W + i*W +j]
        jj = j - D[1][k*H*W + i*W +j]
        kk = k - D[2][k*H*W + i*W +j]

        AEs_new[n] = abs( moving[kk][ii][jj] - fixed[k][i][j] )
        if AEs[n] > AEs_new[n]: c += 1

        #print(i, j, k, ii, jj, kk, AEs[n], AEs_new[n])
        new_csv[n][0] = n + 1 # index starts t 1
        new_csv[n][1] = ii
        new_csv[n][2] = jj - 240
        new_csv[n][3] = kk

    MAE_before = np.median(AEs)
    MAE_after  = np.median(AEs_new)

    r = c / N

    print("MAE_before:", MAE_before, "MAE_after:", MAE_after, "Robustness:", r)

    # write to csv
    print("saving csv file...")
    with open(csv_file_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("Landmark", "X", "Y", "Z"))
        writer.writerows(new_csv)

    return r

def computeJacobiDeterminant(D, H, W, C, nii_file_name):
    n = 0
    jd = np.zeros((H, W, C))          # JD: HWC

    # - compute JD - #
    for h in range(H):
        for w in range(W):
            for c in range(C):
                idx = c*H*W + h*W + w # D:  CHW

                # compute JD elements

                #          | a11 a12 a13 |
                # det(J) = | a21 a22 a23 |
                #          | a31 a32 a33 |

                # a11 = d(tx)/d(x)
                # a13 = d(tx)/d(z)
                # a33 = d(tz)/d(z)

                A = 0
                F = D[2][idx]*D[0][idx]*D[1][idx]

                jd[h][w][c] = -1*A + F
                #print(jd[c][h][w])

                # collect #negative elements
                if jd[h][w][c] < 0: n += 1

    print("# negative elements:", n)

    save_JD_info(jd, H, W, C, nii_file_name)

    return n
