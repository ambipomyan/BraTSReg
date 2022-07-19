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
sliced input dims(HWC): 240 240 10
saving images...
block radius(HWC): 3 3 3
init search window radius(HWC): 10 10 10
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 994
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 185.53028258681297 f(z): 140.15028236154467 ||AX-Z||: 2436.0 ||Xk+1-Xk|| 2436 sw: 10
iter#: 1 F(Z): 183.62366408109665 f(z): 139.97237920109183 ||AX-Z||: 2388.4133929293007 ||Xk+1-Xk|| 6 sw: 10
iter#: 2 F(Z): 183.61891008913517 f(z): 139.97237920109183 ||AX-Z||: 2388.283201856004 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 259.30187903903425 f(z): 186.14981477428228 ||AX-Z||: 1512.6502797580224 ||Xk+1-Xk|| 968 sw: 5
iter#: 1 F(Z): 260.4999379552901 f(z): 186.20567127037793 ||AX-Z||: 1488.8598606429582 ||Xk+1-Xk|| 22 sw: 5
iter#: 2 F(Z): 260.41494557633996 f(z): 186.20567127037793 ||AX-Z||: 1487.3801802007288 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 357.86954229697585 f(z): 256.14890232495964 ||AX-Z||: 708.1206925614391 ||Xk+1-Xk|| 637 sw: 2
iter#: 1 F(Z): 360.7729169987142 f(z): 261.71394817344844 ||AX-Z||: 613.8670605901782 ||Xk+1-Xk|| 89 sw: 2
iter#: 2 F(Z): 358.80564401112497 f(z): 261.75698403827846 ||AX-Z||: 587.8425551932911 ||Xk+1-Xk|| 4 sw: 2
iter#: 3 F(Z): 358.68400977551937 f(z): 261.75698403827846 ||AX-Z||: 586.5015987940141 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 358.70567073114216 f(z): 261.75698403827846 ||AX-Z||: 293.5409596391325 ||Xk+1-Xk|| 0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 1177
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 303.70937514677644 f(z): 252.2093754503876 ||AX-Z||: 2892.0 ||Xk+1-Xk|| 2892 sw: 10
iter#: 1 F(Z): 301.6153016593307 f(z): 252.19940781779587 ||AX-Z||: 2832.979796900194 ||Xk+1-Xk|| 1 sw: 10
iter#: 2 F(Z): 301.614839034155 f(z): 252.19940781779587 ||AX-Z||: 2832.9589161154254 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 391.27512631192803 f(z): 302.313826601021 ||AX-Z||: 1853.7314143820388 ||Xk+1-Xk|| 1072 sw: 5
iter#: 1 F(Z): 392.89373107999563 f(z): 303.5057990802452 ||AX-Z||: 1820.7548828405684 ||Xk+1-Xk|| 45 sw: 5
iter#: 2 F(Z): 392.77432806044817 f(z): 303.5057990802452 ||AX-Z||: 1817.7664037673908 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 513.4418169744313 f(z): 384.1271025734022 ||AX-Z||: 886.1527727591504 ||Xk+1-Xk|| 778 sw: 2
iter#: 1 F(Z): 513.167021818459 f(z): 390.699108527042 ||AX-Z||: 750.611399700725 ||Xk+1-Xk|| 103 sw: 2
iter#: 2 F(Z): 510.9898361116648 f(z): 390.744158254005 ||AX-Z||: 722.0281804303537 ||Xk+1-Xk|| 4 sw: 2
iter#: 3 F(Z): 510.8777067884803 f(z): 390.744158254005 ||AX-Z||: 720.695640705217 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 510.80734100192785 f(z): 390.744158254005 ||AX-Z||: 360.3684168295991 ||Xk+1-Xk|| 0 sw: 1
evaluation:
MAE: before: 23.794024305555556 after: 18.490666666666666
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
