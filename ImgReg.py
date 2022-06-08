import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import random
import math

from kNN import kNN
from MLS import mls

# find data path for original scan and fllowing scan
orignial  = os.path.join(data_path, '/home/kyan2/Desktop/BraTSReg/BraTSReg_001_00_0000_t1.nii.gz')
following = os.path.join(data_path, '/home/kyan2/Desktop/BraTSReg/BraTSReg_001_01_0106_t1.nii.gz')

# load image
fixed_img  = nib.load(orignial)
moving_img = nib.load(following)
fixed_data  = fixed_img.get_fdata()
moving_data = moving_img.get_fdata()

# check image shape
print("inputs(HWC):", fixed_data.shape[0], fixed_data.shape[1], fixed_data.shape[2])

window_x = 8
window_y = 8
window_z = 5

stride = 1
pad = 1

alpha = 1.0
mu    = 8.0
N_x   = round((fixed_data.shape[0]-2*pad) / window_x);
N_y   = round((fixed_data.shape[1]-2*pad) / window_y);
N_z   = round((fixed_data.shape[2]-2*pad) / window_z);
N = N_x * N_y * N_z

# displacement field d
d = np.zeros((3,N), dtype=int)
# matrix A
A = np.eye(N)
# auxiliary variables z
z = np.zeros((3,N)) # init guess: 0.0

# dart throwing
n_throws = 100

# k-NN
knn = 31

print("settings: alpha =", alpha, ", mu =", mu, ", window =", window_x, "x", window_y, "x", window_z, ", grid = ", N_x, "x", N_y, "x", N_z, ", # of estimations = ", n_throws)

# block coordinate descent
error = 0.0
dist = np.zeros((1,1))
reg  = np.zeros((1,1))
for iters in range(n_throws):
    dart_x = random.randint(1, N_x-1)
    dart_y = random.randint(1, N_y-1)
    dart_z = random.randint(1, N_z-1)
    #print("dart(XYZ):", dart_x, dart_y, dart_z)   
    loc = dart_z*N_y*N_x + dart_y*N_x + dart_x
    block_size = window_x*window_y*window_z
    
    h = kNN(fixed_data, moving_data, d, window_x, window_y, window_z, dart_x, dart_y, dart_z, loc, knn, N, A, alpha, mu, dist, reg)
    
    error += float(dist[0])
    
    mls()
    
print("error:", error/n_throws)

# write displacement field to file
with open('weights', 'w') as f:
    for item in d:
        f.write("%s\n" % item)

