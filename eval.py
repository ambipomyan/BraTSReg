import numpy as np
import math
import random

def computeMAE(img1, img2, H, W, C):
    res = 0
    for i in range(H):
        for j in range(W):
            for k in range(C):
                res += abs( img2[k][i][j] - img1[k][i][j] )

    res = res / (H*W*C)

    return res
