# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
- performance and tuning

## dependency
numpy  
nibabel  
cv2  
csv  

## parallelism related
CUDA 11.4  
Numba 0.50.1( llvmlite will be included when installing Numba )   
colorama 0.4.5( >=0.3.9 is fine )  

## sample run: case01, t1
```
input dims(HWC): 240 240 155
sliced input dims(HWC): 240 240 155
saving images...
block radius(HWC): 3 3 3
init search window radius(HWC): 10 10 10
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 8713
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1569.5136807691306 f(z): 1285.8736819848418 ||AX-Z||: 18644.0 ||Xk+1-Xk|| 18644 sw: 10
iter#: 1 F(Z): 1558.4898519404233 f(z): 1285.2075778990984 ||AX-Z||: 18295.331385508387 ||Xk+1-Xk|| 19 sw: 10
iter#: 2 F(Z): 1558.4733152408153 f(z): 1285.2075778990984 ||AX-Z||: 18294.960979254163 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 2062.976045202464 f(z): 1539.771060644649 ||AX-Z||: 11990.788341430396 ||Xk+1-Xk|| 6607 sw: 5
iter#: 1 F(Z): 2069.364585412666 f(z): 1543.1424655681476 ||AX-Z||: 11713.304159488413 ||Xk+1-Xk|| 176 sw: 5
iter#: 2 F(Z): 2068.9897761680186 f(z): 1542.9604161446914 ||AX-Z||: 11705.12192781529 ||Xk+1-Xk|| 9 sw: 5
iter#: 3 F(Z): 2068.954970518127 f(z): 1542.9604161446914 ||AX-Z||: 11704.455135384309 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 2872.5078641446307 f(z): 1990.4924459131435 ||AX-Z||: 6037.401976547984 ||Xk+1-Xk|| 4983 sw: 2
iter#: 1 F(Z): 2838.0228897128254 f(z): 2012.1940196575597 ||AX-Z||: 4979.700196072037 ||Xk+1-Xk|| 314 sw: 2
iter#: 2 F(Z): 2832.2760566202924 f(z): 2012.6877652378753 ||AX-Z||: 4894.232257386346 ||Xk+1-Xk|| 29 sw: 2
iter#: 3 F(Z): 2831.3600742956623 f(z): 2012.6877652378753 ||AX-Z||: 4884.556231987027 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 2831.064562184736 f(z): 2012.6877652378753 ||AX-Z||: 2443.1171592378473 ||Xk+1-Xk|| 0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 8148
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1958.0785385202616 f(z): 1650.6385378492996 ||AX-Z||: 19500.0 ||Xk+1-Xk|| 19500 sw: 10
iter#: 1 F(Z): 1946.1236632745713 f(z): 1650.0222153393552 ||AX-Z||: 19129.3833909532 ||Xk+1-Xk|| 20 sw: 10
iter#: 2 F(Z): 1946.1078083831817 f(z): 1650.0222153393552 ||AX-Z||: 19129.030131429558 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 2501.901967389509 f(z): 1913.1148672690615 ||AX-Z||: 13073.267865116135 ||Xk+1-Xk|| 6404 sw: 5
iter#: 1 F(Z): 2510.5742672849447 f(z): 1917.0549339642748 ||AX-Z||: 12859.23957602299 ||Xk+1-Xk|| 176 sw: 5
iter#: 2 F(Z): 2510.2370999660343 f(z): 1916.9656828036532 ||AX-Z||: 12851.31413457373 ||Xk+1-Xk|| 4 sw: 5
iter#: 3 F(Z): 2510.219755301252 f(z): 1916.9656828036532 ||AX-Z||: 12851.017720235031 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 3392.075841953978 f(z): 2425.121703351848 ||AX-Z||: 6566.77241148848 ||Xk+1-Xk|| 5269 sw: 2
iter#: 1 F(Z): 3377.284993296489 f(z): 2457.5509262708947 ||AX-Z||: 5559.316289584579 ||Xk+1-Xk|| 464 sw: 2
iter#: 2 F(Z): 3368.3618007972836 f(z): 2458.1445118216798 ||AX-Z||: 5431.690730918626 ||Xk+1-Xk|| 35 sw: 2
iter#: 3 F(Z): 3367.278685728088 f(z): 2458.1445118216798 ||AX-Z||: 5420.022862311624 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 3366.7226494345814 f(z): 2458.1445118216798 ||AX-Z||: 2710.0390606981973 ||Xk+1-Xk|| 0 sw: 1
evaluation:
MAE: before: 13.242365815412187 after: 10.484244735663083
Robustness: 0.16666666666666666
```
## visualization(image slice in the middle, 240 x 240 x 1, mm)
#### inputs:
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
#### itermediate results:
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
darts image:  
![darts image](https://github.com/ambipomyan/BraTSReg/blob/main/darts.jpg)  
#### result:
![pred image](https://github.com/ambipomyan/BraTSReg/blob/main/pred.jpg)  
