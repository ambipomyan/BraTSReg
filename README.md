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

## sample run
```
input dims(HWC): 240 240 155
sliced input dims(HWC): 240 240 155
saving images...
block radius(HWC): 3 3 1
init search window radius(HWC): 10 10 10
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 17572
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 3375.8387296106666 f(z): 2768.388728563208 ||AX-Z||: 39061.0 ||Xk+1-Xk|| 39061 sw: 10
iter#: 1 F(Z): 3352.234477496706 f(z): 2767.286936101038 ||AX-Z||: 38323.87476379344 ||Xk+1-Xk|| 33 sw: 10
iter#: 2 F(Z): 3352.206273649819 f(z): 2767.286936101038 ||AX-Z||: 38323.22772046629 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 4428.171825444326 f(z): 3300.694411179982 ||AX-Z||: 25449.865407727688 ||Xk+1-Xk|| 14412 sw: 5
iter#: 1 F(Z): 4436.369940310717 f(z): 3307.2358172452077 ||AX-Z||: 24868.516136512313 ||Xk+1-Xk|| 439 sw: 5
iter#: 2 F(Z): 4435.058781962842 f(z): 3306.7250109976158 ||AX-Z||: 24846.614586237225 ||Xk+1-Xk|| 15 sw: 5
iter#: 3 F(Z): 4434.978413678706 f(z): 3306.7250109976158 ||AX-Z||: 24845.50540822428 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 6175.117905807681 f(z): 4264.070058178157 ||AX-Z||: 12893.328550739865 ||Xk+1-Xk|| 10119 sw: 2
iter#: 1 F(Z): 6118.836193723604 f(z): 4310.355882424861 ||AX-Z||: 10847.662732944913 ||Xk+1-Xk|| 800 sw: 2
iter#: 2 F(Z): 6102.134510204196 f(z): 4310.705985266715 ||AX-Z||: 10622.311306521597 ||Xk+1-Xk|| 52 sw: 2
iter#: 3 F(Z): 6100.245827142149 f(z): 4310.705985266715 ||AX-Z||: 10604.975643663034 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 6099.718608601019 f(z): 4310.705985266715 ||AX-Z||: 5302.6518448821 ||Xk+1-Xk|| 0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 16479
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 4387.231446491554 f(z): 3725.9614470638335 ||AX-Z||: 41285.0 ||Xk+1-Xk|| 41285 sw: 10
iter#: 1 F(Z): 4361.533225623891 f(z): 3724.402444511652 ||AX-Z||: 40518.14491240258 ||Xk+1-Xk|| 43 sw: 10
iter#: 2 F(Z): 4361.492224963382 f(z): 3724.402444511652 ||AX-Z||: 40517.30092113168 ||Xk+1-Xk|| 0 sw: 10
iter#: 0 F(Z): 5529.616666525602 f(z): 4318.799547906965 ||AX-Z||: 26978.906528337266 ||Xk+1-Xk|| 15236 sw: 5
iter#: 1 F(Z): 5542.880647464655 f(z): 4326.132888395339 ||AX-Z||: 26425.91680321659 ||Xk+1-Xk|| 510 sw: 5
iter#: 2 F(Z): 5541.569569253363 f(z): 4325.609844643623 ||AX-Z||: 26401.853363496313 ||Xk+1-Xk|| 10 sw: 5
iter#: 3 F(Z): 5541.5031397091225 f(z): 4325.526050705463 ||AX-Z||: 26403.111594435515 ||Xk+1-Xk|| 2 sw: 5
iter#: 4 F(Z): 5541.491726514883 f(z): 4325.526050705463 ||AX-Z||: 26402.96344333107 ||Xk+1-Xk|| 0 sw: 5
iter#: 0 F(Z): 7382.46322133299 f(z): 5383.522496064194 ||AX-Z||: 13505.489010831423 ||Xk+1-Xk|| 10869 sw: 2
iter#: 1 F(Z): 7336.5869193011895 f(z): 5442.866819952615 ||AX-Z||: 11386.302580744083 ||Xk+1-Xk|| 1081 sw: 2
iter#: 2 F(Z): 7311.0658836709335 f(z): 5442.722030674107 ||AX-Z||: 11068.89330024546 ||Xk+1-Xk|| 55 sw: 2
iter#: 3 F(Z): 7308.986796007492 f(z): 5442.722030674107 ||AX-Z||: 11050.592104974905 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 7308.539625728503 f(z): 5442.722030674107 ||AX-Z||: 5525.667662687818 ||Xk+1-Xk|| 0 sw: 1
MAE: before: 13.242365815412187 after: 13.242498655913979
```
## visualization(slice#100)
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
