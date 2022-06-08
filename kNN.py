import math
import numpy as np

def distance(dst, src, d, wx, wy, wz, dart_x, dart_y, dart_z, loc, s_x, s_y, s_z):
    dist = 0
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                dist += (src[s_x*wx+i+d[0][loc]][s_y*wy+j+d[1][loc]][s_z*wz+k+d[2][loc]] - dst[dart_x*wx+i][dart_y*wy+j][dart_z*wz+k])**2
    
    dist = math.sqrt(dist)
    
    return dist

def regulation(A, d, alpha): # TODO
    reg = 0
    
    return reg

def search(dist, d, dst, src, wx, wy, wz, dart_x, dart_y, dart_z, loc, N):
    dist_new = 0.0
    d0 = np.zeros((3,N), dtype=int)
    for idx in range(-wx, wx):
           for idy in range(-wy, wy):
               for idz in range(-wz, wz):
                   dist_new = distance(dst, src, d, wx, wy, wz, dart_x, dart_y, dart_z, loc, idx, idy, idz)
                   if dist_new < dist[0]:
                       dist[0] = dist_new
                       d[0][loc] = idx - dart_x
                       d[1][loc] = idy - dart_y
                       d[2][loc] = idz - dart_z
                     
    return [idx, idy, idz]

# block matching for QPDIR
def kNN(dst, src, d, wx, wy, wz, dart_x, dart_y, dart_z, loc, N, knn, A, alpha, mu, dist, reg):
    h = 0
    
    # compute distance, i.e. function F
    dist[0] = distance(dst, src, d, wx, wy, wz, dart_x, dart_y, dart_z, loc, dart_x, dart_y, dart_z)
    
    # compute QP
    reg[0] = regulation(A, d, alpha)
    
    # block matching: exhausted search
    idx, idy, idz = search(dist, d, dst, src, wx, wy, wz, dart_x, dart_y, dart_z, loc, N)
    
    return h
