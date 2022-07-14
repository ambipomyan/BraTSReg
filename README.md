# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
performance and tuning

## dependency
numpy  
nibabel  
cv2  

## parallelism related
CUDA 11.4  
Numba 0.50.1( llvmlite will be included when installing Numba )   
colorama 0.4.5( >=0.3.9 is fine )  

## sample run
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
# of list points for mls: 1233
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1036.9468700915932 f(z): 1031.9513145360384 ||AX-Z||: 550 ||Xk+1-Xk|| 550.0 sw: 15
iter#: 1 F(Z): 1036.8613921041417 f(z): 1031.9513145360384 ||AX-Z||: 0 ||Xk+1-Xk|| 545.4075891689004 sw: 15
iter#: 0 F(Z): 1043.9844701022757 f(z): 1036.2158732228283 ||AX-Z||: 186 ||Xk+1-Xk|| 377.9707568232298 sw: 8
iter#: 1 F(Z): 1043.9970438170578 f(z): 1036.1788786609973 ||AX-Z||: 4 ||Xk+1-Xk|| 376.3181275506216 sw: 8
iter#: 2 F(Z): 1043.9932795837037 f(z): 1036.1788786609973 ||AX-Z||: 0 ||Xk+1-Xk|| 376.1966385052592 sw: 8
iter#: 0 F(Z): 1056.0616832030635 f(z): 1043.2215661259022 ||AX-Z||: 147 ||Xk+1-Xk|| 229.29604542823407 sw: 4
iter#: 1 F(Z): 1056.4278147170526 f(z): 1043.2585145738742 ||AX-Z||: 1 ||Xk+1-Xk|| 222.27641147176433 sw: 4
iter#: 2 F(Z): 1056.4270580752911 f(z): 1043.2585145738742 ||AX-Z||: 0 ||Xk+1-Xk|| 222.16928933683724 sw: 4
iter#: 0 F(Z): 1070.2692791370048 f(z): 1047.8831220305706 ||AX-Z||: 55 ||Xk+1-Xk|| 143.14865435060904 sw: 2
iter#: 1 F(Z): 1069.557356383502 f(z): 1047.8958406347742 ||AX-Z||: 1 ||Xk+1-Xk|| 129.805020458751 sw: 2
iter#: 2 F(Z): 1069.5295734345978 f(z): 1047.8958406347742 ||AX-Z||: 0 ||Xk+1-Xk|| 129.471666440911 sw: 2
iter#: 0 F(Z): 1069.5246858762796 f(z): 1047.8958406347742 ||AX-Z||: 0 ||Xk+1-Xk|| 64.7501793587896 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 1384
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1161.525368474947 f(z): 1157.218701808281 ||AX-Z||: 533 ||Xk+1-Xk|| 533.0 sw: 15
iter#: 1 F(Z): 1161.4509841788195 f(z): 1157.218701808281 ||AX-Z||: 0 ||Xk+1-Xk|| 528.4595560984874 sw: 15
iter#: 0 F(Z): 1168.3026965643994 f(z): 1160.530586968764 ||AX-Z||: 153 ||Xk+1-Xk|| 403.97746392539733 sw: 8
iter#: 1 F(Z): 1168.3891307998856 f(z): 1160.5444423492129 ||AX-Z||: 4 ||Xk+1-Xk|| 401.4588543045995 sw: 8
iter#: 2 F(Z): 1168.3853149393926 f(z): 1160.5444423492129 ||AX-Z||: 0 ||Xk+1-Xk|| 401.33754770078275 sw: 8
iter#: 0 F(Z): 1181.0081242550375 f(z): 1168.0403571805728 ||AX-Z||: 158 ||Xk+1-Xk|| 242.22122976972716 sw: 4
iter#: 1 F(Z): 1181.3224432859192 f(z): 1168.0403571805728 ||AX-Z||: 0 ||Xk+1-Xk|| 233.3291012281793 sw: 4
iter#: 0 F(Z): 1196.637415912591 f(z): 1172.4009076614823 ||AX-Z||: 56 ||Xk+1-Xk|| 154.63729067909048 sw: 2
iter#: 1 F(Z): 1195.406664965621 f(z): 1172.4009076614823 ||AX-Z||: 0 ||Xk+1-Xk|| 137.8486593774035 sw: 2
iter#: 0 F(Z): 1195.40395233432 f(z): 1172.4009076614823 ||AX-Z||: 0 ||Xk+1-Xk|| 68.95942049275676 sw: 1

```
## visualization
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
