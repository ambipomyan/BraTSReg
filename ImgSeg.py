import numpy as np
import math
import random
import cv2


def convertToMax(data, H, W, C, Max):
    max_tmp = np.amax(data)
    min_tmp = 0
    #print("Max:", max_tmp, "Min:", min_tmp)
    tmp = np.zeros((C, H, W), dtype=int)
    for k in range(C):
        for i in range(H):
            for j in range(W):
                tmp[k][i][j] = (data[k][i][j] - min_tmp)/max_tmp*Max

    return tmp

def createMask(data, H, W, C):
    # range: 0 ~ 255
    mask_data = convertToMax(data, H, W, C, 255)
    # raw segmentation based on intensity
    for k in range(C):
        for i in range(H):
            for j in range(W):
                if mask_data[k][i][j] > 120: # use small value for faster tests
                    mask_data[k][i][j] = 2
                elif mask_data[k][i][j] > 0:
                    mask_data[k][i][j] = 1
                else:
                    mask_data[k][i][j] = 0

    return mask_data

def getDisplacementField(d, d_ws, L, H, W, C, dpx, dpy, dpz):
    D = np.zeros((3, H*W*C), dtype=int)

    for l in range(L):
        i_l = d_ws[0][l]
        j_l = d_ws[1][l]
        k_l = d_ws[2][l]

        for c in range(-dpz, dpz+1):
            for h in range(-dpx, dpx+1):
                for w in range(-dpy, dpy+1):
                    k = k_l + c
                    i = i_l + h
                    j = j_l + w

                    if k < 0: k = 0
                    if k >= C: k = C - 1

                    D[2][k*H*W + i*W + j] = d[2][l]
                    D[0][k*H*W + i*W + j] = d[0][l]
                    D[1][k*H*W + i*W + j] = d[1][l]

                    if k + d[2][l] < 0:  D[2][k*H*W + i*W + j] = 0 - k
                    if k + d[2][l] >= C: D[2][k*H*W + i*W + j] = C - 1 - k

    return D

def genPred(D, moving_data, H, W, C):
    pred_data = np.zeros((C, H, W), dtype=int)

    for k in range(C):
        for i in range(H):
            for j in range(W):
                idx = k*H*W + i*W + j
                t_k = k + D[2][idx]
                t_i = i + D[0][idx]
                t_j = j + D[1][idx]

                pred_data[t_k][t_i][t_j] = moving_data[k][i][j]

    return pred_data

# ----- visualization ----- #

def saveImg(img_data, H, W, C, file_name, scale):
    max_tmp = np.amax(img_data)
    min_tmp = 0
    tmp = np.zeros((H, W), dtype=int)
    sec = int( round(C/2) )
    for i in range(H):
        for j in range(W):
            tmp[i][j] = (img_data[sec][i][j] - min_tmp)/max_tmp*255*scale

    cv2.imwrite(file_name, tmp)

    return 0
