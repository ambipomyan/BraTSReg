# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
- performance and tuning

## dependency
numpy  
nibabel  
cv2  

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
# of list points for mls: 8731
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1576.885083321482 f(z): 1287.075083765667 ||AX-Z||: 18845.0 ||Xk+1-Xk|| 18845 sw: 10
iter#: 1 F(Z): 1565.6103890212253 f(z): 1286.1722945156507 ||AX-Z||: 18498.034712437777 ||Xk+1-Xk|| 23 sw: 10
iter#: 2 F(Z): 1565.588393590413 f(z): 1286.1722945156507 ||AX-Z||: 18497.584362943504 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 2075.691760668531 f(z): 1545.9937102301046 ||AX-Z||: 12164.806880235288 ||Xk+1-Xk|| 6655 sw: 5
iter#: 1 F(Z): 2082.657490156591 f(z): 1547.3879826841876 ||AX-Z||: 11925.73383352268 ||Xk+1-Xk|| 131 sw: 5
iter#: 2 F(Z): 2082.3437839895487 f(z): 1547.3368321238086 ||AX-Z||: 11918.982807345727 ||Xk+1-Xk|| 1 sw: 5
iter#: 3 F(Z): 2082.338119406253 f(z): 1547.3368321238086 ||AX-Z||: 11918.909217029473 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 2900.533151999116 f(z): 2001.7014489425346 ||AX-Z||: 6148.948139189558 ||Xk+1-Xk|| 5052 sw: 2
iter#: 1 F(Z): 2864.8601536639035 f(z): 2024.728374841623 ||AX-Z||: 5071.859137631347 ||Xk+1-Xk|| 359 sw: 2
iter#: 2 F(Z): 2857.7291668727994 f(z): 2025.0365164009854 ||AX-Z||: 4972.490963602111 ||Xk+1-Xk|| 30 sw: 2
iter#: 3 F(Z): 2856.6826449725777 f(z): 2025.0365164009854 ||AX-Z||: 4962.516230794449 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 2856.349227981642 f(z): 2025.0365164009854 ||AX-Z||: 2481.980357032672 ||Xk+1-Xk|| 0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 8159
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1940.4048582091928 f(z): 1634.2648560320958 ||AX-Z||: 19524.0 ||Xk+1-Xk|| 19524 sw: 10
iter#: 1 F(Z): 1928.502920538187 f(z): 1633.6786331823096 ||AX-Z||: 19155.940861094332 ||Xk+1-Xk|| 19 sw: 10
iter#: 2 F(Z): 1928.4878321364522 f(z): 1633.6786331823096 ||AX-Z||: 19155.567773290994 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 2482.2079701647162 f(z): 1891.3515061177313 ||AX-Z||: 13112.876918766806 ||Xk+1-Xk|| 6349 sw: 5
iter#: 1 F(Z): 2491.3394899852574 f(z): 1895.1449312753975 ||AX-Z||: 12912.291555155727 ||Xk+1-Xk|| 167 sw: 5
iter#: 2 F(Z): 2491.049539161846 f(z): 1895.1449312753975 ||AX-Z||: 12902.142662259044 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 3371.030663024634 f(z): 2413.6840292792767 ||AX-Z||: 6544.8068424555395 ||Xk+1-Xk|| 5417 sw: 2
iter#: 1 F(Z): 3352.8217376638204 f(z): 2443.674747450277 ||AX-Z||: 5504.199419851459 ||Xk+1-Xk|| 433 sw: 2
iter#: 2 F(Z): 3344.1277192421257 f(z): 2444.108734888956 ||AX-Z||: 5384.489884816315 ||Xk+1-Xk|| 38 sw: 2
iter#: 3 F(Z): 3343.0721992161125 f(z): 2444.108734888956 ||AX-Z||: 5371.821968090241 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 3342.6301253437996 f(z): 2444.108734888956 ||AX-Z||: 2686.168951373385 ||Xk+1-Xk|| 0 sw: 1
MAE: before: 13.242365815412187 after: 10.461548275089605
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
