import numpy as np
import math
import random

def distance(dst, src, wx, wy, wz, dart_x, dart_y, dart_z, s_x, s_y, s_z):
    dist = 0
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                dist += (src[s_x+i][s_y+j][s_z+k] - dst[dart_x*wx+i][dart_y*wy+j][dart_z*wz+k])**2
                
    
    dist = math.sqrt(dist)
    
    return dist

def search_by_block(dist, d, dst, src, wx, wy, wz, dart_x, dart_y, dart_z, loc, N):
    for idx in range(-wx, wx):
        for idy in range(-wy, wy):
            for idz in range(-wz, wz):
                dist[1] = distance(dst, src, wx, wy, wz, dart_x, dart_y, dart_z, idx+dart_x*wx, idy+dart_y*wy, idz+dart_z*wz)
                if dist[1] < dist[0]:
                    dist[0] = dist[1]
                    d[0][loc] = -idx
                    d[1][loc] = -idy
                    d[2][loc] = -idz

    return idx, idy, idz

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
