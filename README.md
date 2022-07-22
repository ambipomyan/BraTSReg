# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
- Check dev_S data copy!!!!!!!!!!!
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
# of list points for mls: 8737
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1575.2923741620034 f(z): 1284.7323747412302 ||AX-Z||: 18872.0 ||Xk+1-Xk|| 18872 sw: 10
iter#: 1 F(Z): 1563.9907189942896 f(z): 1283.916764023248 ||AX-Z||: 18525.576491685933 ||Xk+1-Xk|| 24 sw: 10
iter#: 2 F(Z): 1563.9698498807847 f(z): 1283.916764023248 ||AX-Z||: 18525.105973628943 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 2076.5108958929777 f(z): 1540.9686431996524 ||AX-Z||: 12212.46528598536 ||Xk+1-Xk|| 6722 sw: 5
iter#: 1 F(Z): 2082.038699954748 f(z): 1544.9853965006769 ||AX-Z||: 11922.77284590971 ||Xk+1-Xk|| 168 sw: 5
iter#: 2 F(Z): 2081.7230022829026 f(z): 1545.0360090099275 ||AX-Z||: 11912.55145621295 ||Xk+1-Xk|| 4 sw: 5
iter#: 3 F(Z): 2081.716637259349 f(z): 1545.0360090099275 ||AX-Z||: 11912.403236871669 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 2893.9910803362727 f(z): 2006.5675192121416 ||AX-Z||: 6098.717181283474 ||Xk+1-Xk|| 5182 sw: 2
iter#: 1 F(Z): 2855.1496459916234 f(z): 2029.8803762737662 ||AX-Z||: 4982.471462698525 ||Xk+1-Xk|| 342 sw: 2
iter#: 2 F(Z): 2849.483670834452 f(z): 2030.4872398916632 ||AX-Z||: 4892.438263760135 ||Xk+1-Xk|| 20 sw: 2
iter#: 3 F(Z): 2848.9282575342804 f(z): 2030.4872398916632 ||AX-Z||: 4885.768824356498 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 2848.522108387202 f(z): 2030.4872398916632 ||AX-Z||: 2443.132995629909 ||Xk+1-Xk|| 0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 8066
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1945.5299744904041 f(z): 1640.3699750378728 ||AX-Z||: 19270.0 ||Xk+1-Xk|| 19270 sw: 10
iter#: 1 F(Z): 1933.665564769879 f(z): 1639.8535796776414 ||AX-Z||: 18909.832019662375 ||Xk+1-Xk|| 20 sw: 10
iter#: 2 F(Z): 1933.6516512446105 f(z): 1639.8535796776414 ||AX-Z||: 18909.440043828377 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 2480.973811868578 f(z): 1898.338155648671 ||AX-Z||: 12914.998350334863 ||Xk+1-Xk|| 6313 sw: 5
iter#: 1 F(Z): 2488.8618830293417 f(z): 1901.1936716055498 ||AX-Z||: 12717.292928455825 ||Xk+1-Xk|| 162 sw: 5
iter#: 2 F(Z): 2488.488852996379 f(z): 1901.1427489137277 ||AX-Z||: 12708.365772444384 ||Xk+1-Xk|| 1 sw: 5
iter#: 3 F(Z): 2488.483147881925 f(z): 1901.1427489137277 ||AX-Z||: 12708.291714177543 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 3358.6995367389172 f(z): 2414.2854505563155 ||AX-Z||: 6451.552777184093 ||Xk+1-Xk|| 5314 sw: 2
iter#: 1 F(Z): 3342.090235900134 f(z): 2448.0940339872614 ||AX-Z||: 5417.773872067409 ||Xk+1-Xk|| 487 sw: 2
iter#: 2 F(Z): 3331.6197627820075 f(z): 2448.6652311990038 ||AX-Z||: 5279.761723119429 ||Xk+1-Xk|| 33 sw: 2
iter#: 3 F(Z): 3330.6389420684427 f(z): 2448.350578696467 ||AX-Z||: 5269.760152150412 ||Xk+1-Xk|| 1 sw: 2
iter#: 4 F(Z): 3330.5000563468784 f(z): 2448.350578696467 ||AX-Z||: 5269.426820125841 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 3329.9433409590274 f(z): 2448.350578696467 ||AX-Z||: 2634.872655264619 ||Xk+1-Xk|| 0 sw: 1
===========
evaluation:
MAE: before: 13.242365815412187 after: 10.514192316308243
Robustness: 0 / 6
Jacobian Determinat: #negative elements:  894 / 8928000 Max and Mean vals: 8.0 0.9239940636200716
```
## visualization(image slice in the middle, 240 x 240 x 1, mm)
#### inputs:
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
#### itermediate result:
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
#### result:
![pred image](https://github.com/ambipomyan/BraTSReg/blob/main/pred.jpg)  
