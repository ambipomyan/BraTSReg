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

def genPredImg(d, moving_data, H, W, C):
    pred_data = np.zeros((C, H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            for k in range(C):
                ID = k*H*W + i*W + j

                #print("i, j, k, ID, d[0], d[1], d[2]:", i,        j,        k,       \
                                                        #ID,                          \
                                                        #d[0][ID], d[1][ID], d[2][ID])

                pred_data[k][i][j] = moving_data[k + d[2][ID]][i + d[0][ID]][j + d[1][ID]]

    return pred_data
