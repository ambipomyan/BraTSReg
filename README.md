# BraTSReg
Prepared for MICCAI 2022 challenge

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
