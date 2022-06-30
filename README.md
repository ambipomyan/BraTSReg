# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
```
input dims(HWC): 240 240 155
sliced input dims(HWC): 240 240 3
saving images...
dtype: int16
block radius(HWC): 3 3 1
init search window radius(HWC): 15 15 1
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 1276
voxel dim by mm (HWC): 1 1 1
rTr: 0.0
rTr: 0.0
rTr: 0.0
S[0], S[1], S[2], pid: 110 218 2 0
src[0], src[1], src[2]: 110 218 2
Y[0], Y[1], Y[2], pid: 0.0 0.0 0.0 0
tar[0], tar[1], tar[2]: 110 218 2
d[0], d[1], d[2]: 0.0 0.0 0.0
tk, ti, tj: 1 95 203
tk, ti, tj: 2 104 211
tk, ti, tj: 2 111 219
tk, ti, tj: 2 119 227
tk, ti, tj: 2 98 204
tk, ti, tj: 2 106 212
tk, ti, tj: 2 113 220
tk, ti, tj: 2 121 228
tk, ti, tj: 3 100 205
Traceback (most recent call last):
  File "ImgReg.py", line 177, in <module>
    objVal, ccVal = updateDisplacementField(fixed_data, moving_data, F, I, z_ws, Z, Y, L, localVals, mu, sx, sy, sz, rx, ry, rz)
  File "/home/kyan2/Desktop/BraTSReg/QPDIR.py", line 79, in updateDisplacementField
    searchMin(fixed, moving, count, F, I, S, Z, Y, L, mu, sx, sy, sz, rx, ry, rz)
  File "/home/kyan2/Desktop/BraTSReg/QPDIR.py", line 171, in searchMin
    q = fixed[k + tk][i + ti][j + tj]
IndexError: index 3 is out of bounds for axis 0 with size 3

```

## dependency
numpy  
nibabel  
cv2  

## sample run
```
input dims(HWC): 240 240 155
sliced input dims(HWC): 240 240 3
saving images...
dtype: int16
block radius(HWC): 3 3 1
init search window radius(HWC): 15 15 15
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 1179
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 15
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 8
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 4
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 2
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 1425
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 15
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 8
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 4
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 2
iter#: 0 F(Z): 0.0 f(z): 0.0 ||AX-Z||: 0.0 ||Xk+1-Xk|| 0.0 sw: 1

```
## visualization
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
