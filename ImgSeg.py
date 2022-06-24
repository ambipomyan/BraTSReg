import numpy as np
import math
import random

def convert2Int8(data, H, W, C):
    max_tmp = np.amax(data)
    min_tmp = 0
    #print("Max:", max_tmp, "Min:", min_tmp)
    tmp = (data - min_tmp)/max_tmp*255 + 0

    return tmp

def createMask(data, H, W, C):
    # range: 0 ~ 255
    mask_data = convert2Int8(data, H, W, C)

    # raw segmentation based on intensity
    for i in range(H):
        for j in range(W):
            for k in range(C):
                if mask_data[i][j][k] > 20: # use small value for faster tests
                    mask_data[i][j][k] = 2
                elif mask_data[i][j][k] > 0:
                    mask_data[i][j][k] = 1

    mask_data = mask_data.reshape(H*W*C)

    return mask_data
