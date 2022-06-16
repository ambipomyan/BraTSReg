import numpy as np
import math
import random

from kNN import distance

def dart_throw(N_x, N_y, N_z, stride, padding):
    dart_x = random.randint(0, N_x-2*padding)
    dart_y = random.randint(0, N_y-2*padding)
    dart_z = random.randint(0, N_z-2*padding)   
    loc = dart_z*N_y*N_x + dart_y*N_x + dart_x

    return dart_x, dart_y, dart_z, loc

def get_grid(H, W, C, wx, wy, wz, stride, padding):
    N_x   = round((H-2*padding) / wx);
    N_y   = round((W-2*padding) / wy);
    N_z   = round((C-2*padding) / wz);
    N = N_x * N_y * N_z

    return N_x, N_y, N_z, N

# Sampler for warpping moving image to predicted image
def sample_by_block(output_data, input_data, d, loc, i, j ,k, wx, wy, wz):
    for p in range(wx):
        for q in range(wy):
            for r in range(wz):
                x = i*wx + p + d[0][loc]
                y = j*wy + q + d[1][loc]
                z = k*wz + r + d[2][loc]
                output_data[x][y][z] = input_data[i*wx+p][j*wy+q][k*wz+r]

def sample(fixed_data, pred_data, moving_data, d, window_x, window_y, window_z, N_x, N_y, N_z):
    error = 0.0
    for i in range(N_x):
        for j in range(N_y):
            for k in range(N_z):
                loc = k*N_y*N_x + j*N_x + i
                sample_by_block(pred_data, moving_data, d, loc, i, j ,k, window_x, window_y, window_z)
                error += distance(fixed_data, pred_data, window_x, window_y, window_z, i, j, k, i*window_x, j*window_y, k*window_z)

    return error
