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

def genPredImg(d, d_ws, L, moving_data, H, W, C):
    pred_darts = np.zeros((C, H, W), dtype=int)
    pred_data = np.zeros((C, H, W), dtype=int)

    for l in range(L):
        # get src index
        k = d_ws[2][l]
        i = d_ws[0][l]
        j = d_ws[1][l]

        # get displacement field index
        #ID = d_ws[2][l]*H*W + d_ws[0][l]*W + d_ws[0][l]
        t_x = i + d[0][l]
        t_y = j + d[1][l]
        t_z = k + d[2][l]

        if t_x < 0: t_x = 0
        if t_y < 0: t_y = 0
        if t_z < 0: t_z = 0

        if t_x >= H: t_x = H - 1
        if t_y >= W: t_y = W - 1
        if t_z >= C: t_z = C - 1

        #print("i, j, k, ID, d[0], d[1], d[2]:", i, j, k, ID, d[0][ID], d[1][ID], d[2][ID])
        pred_darts[k][i][j] = moving_data[t_z][t_x][t_y]

    for k in range(C):
        for i in range(H):
            for j in range(W):
                if pred_darts[k][i][j] != 0:
                    pred_data[k][i][j] = pred_darts[k][i][j]
                else:
                    pred_data[k][i][j] = moving_data[k][i][j]

    return pred_data, pred_darts
