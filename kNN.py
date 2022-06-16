import math
import numpy as np

def distance(dst, src, wx, wy, wz, dart_x, dart_y, dart_z, s_x, s_y, s_z):
    dist = 0
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                dist += (src[s_x+i][s_y+j][s_z+k] - dst[dart_x*wx+i][dart_y*wy+j][dart_z*wz+k])**2
                
    
    dist = math.sqrt(dist)
    
    return dist

def regulation(A, d, alpha): # TODO
    reg = 0
    
    return reg

def search(dist, d, dst, src, wx, wy, wz, dart_x, dart_y, dart_z, loc, N):
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

# block matching for QPDIR
def kNN(dst, src, d, wx, wy, wz, dart_x, dart_y, dart_z, loc, N, knn, A, alpha, mu, dist, reg, stride, padding):
    h = 0
    
    # compute distance, i.e. function F
    dist[0] = distance(dst, src, wx, wy, wz, dart_x, dart_y, dart_z, dart_x*wx, dart_y*wy, dart_z*wz)
    
    # compute QP
    reg[0] = regulation(A, d, alpha)
    
    # block matching: exhausted search
    idx, idy, idz = search(dist, d, dst, src, wx, wy, wz, dart_x, dart_y, dart_z, loc, N)
    
    return h
