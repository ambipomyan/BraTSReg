import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import random
import math

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

# dart throwing
error = 0
pad = 1
for r in range(100):
    window_size = 24
    dart_x = random.randint(pad+window_size, fixed_data.shape[0]-window_size-pad)
    dart_y = random.randint(pad+window_size, fixed_data.shape[1]-window_size-pad)
    dart_z = random.randint(pad, fixed_data.shape[2]-pad)

    print("dart(XYZ):", dart_x, dart_y, dart_z)

# compute similarity
    for i in range(window_size):
        for j in range(window_size):
            error += (moving_data[dart_x+i][dart_y+j][dart_z] - fixed_data[dart_x+i][dart_y+j][dart_z])**2
error = math.sqrt(error/100)

print("similarity(MSE):", error)
