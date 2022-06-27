import numpy as np
import math
import random
import cv2

def convert2Int8(data, H, W, C):
    max_tmp = np.amax(data)
    min_tmp = 0
    #print("Max:", max_tmp, "Min:", min_tmp)
    tmp = np.zeros((C, H, W), dtype=int)
    for k in range(C):
        for i in range(H):
            for j in range(W):
                tmp[k][i][j] = (data[k][i][j] - min_tmp)/max_tmp*255 + 0

    return tmp

def createMask(data, H, W, C):
    # range: 0 ~ 255
    mask_data = convert2Int8(data, H, W, C)

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

def saveImg(img_data, H, W, file_name, scale):
    tmp = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            tmp[i][j] = img_data[0][i][j]*scale

    cv2.imwrite(file_name, tmp)

    return 0
