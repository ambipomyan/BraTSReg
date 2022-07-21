import numpy as np
import math
import random
import cv2

def convertTo255(data, H, W, C):
    max_tmp = np.amax(data)
    min_tmp = 0
    #print("Max:", max_tmp, "Min:", min_tmp)
    tmp = np.zeros((C, H, W), dtype=int)
    for k in range(C):
        for i in range(H):
            for j in range(W):
                tmp[k][i][j] = (data[k][i][j] - min_tmp)/max_tmp*255

    return tmp

def createMask(data, H, W, C):
    # range: 0 ~ 255
    mask_data = convertTo255(data, H, W, C)

    # raw segmentation based on intensity
    for k in range(C):
        for i in range(H):
            for j in range(W):
                if mask_data[k][i][j] > 160: # use small value for faster tests
                    mask_data[k][i][j] = 2
                elif mask_data[k][i][j] > 0:
                    mask_data[k][i][j] = 1
                else:
                    mask_data[k][i][j] = 0

    return mask_data


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

def genFullPred(d, d_ws, L, moving_data, H, W, C, dpx, dpy, dpz):
    pred_darts = np.zeros((C, H, W), dtype=int)

    for l in range(L):
        i_l = d_ws[0][l]
        j_l = d_ws[1][l]
        k_l = d_ws[2][l]

        t_i_l = i_l + d[0][l]
        t_j_l = j_l + d[1][l]
        t_k_l = k_l + d[2][l]

        for c in range(-dpz, dpz+1):
            for h in range(-dpx, dpx+1):
                for w in range(-dpy, dpy+1):
                    k = k_l + c
                    i = i_l + h
                    j = j_l + w

                    t_k = t_k_l + c
                    t_i = t_i_l + h
                    t_j = t_j_l + w

                    if k < 0: k = 0
                    if i < 0: i = 0
                    if j < 0: j = 0
                    if k >= C: k = C - 1
                    if i >= H: i = H - 1
                    if j >= W: j = W - 1

                    if t_k < 0: t_k = 0
                    if t_i < 0: t_i = 0
                    if t_j < 0: t_j = 0
                    if t_k >= C: t_k = C - 1
                    if t_i >= H: t_i = H - 1
                    if t_j >= W: t_j = W - 1

                    pred_darts[t_k][t_i][t_j] = moving_data[k][i][j]

    return pred_darts

def genPredImg(pred_darts, moving_data, H, W, C):
    pred_data = np.zeros((C, H, W), dtype=int)

    for k in range(C):
        for i in range(H):
            for j in range(W):
                if pred_darts[k][i][j] != 0:
                    pred_data[k][i][j] = pred_darts[k][i][j]
                else:
                    pred_data[k][i][j] = moving_data[k][i][j]

    return pred_data
