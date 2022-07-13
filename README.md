# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
performance and tuning

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
init search window radius(HWC): 15 15 1
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 1244
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1277.0786359261822 f(z): 1182.1241839780862 ||AX-Z||: 38564 ||Xk+1-Xk|| 38564.0 sw: 15
iter#: 1 F(Z): 1317.354470008282 f(z): 1188.4622455602935 ||AX-Z||: 36074 ||Xk+1-Xk|| 38311.45415979206 sw: 15
iter#: 2 F(Z): 1289.32690926474 f(z): 1176.8958936152194 ||AX-Z||: 33584 ||Xk+1-Xk|| 38054.36275608766 sw: 15
iter#: 3 F(Z): 1315.3810723589154 f(z): 1193.2720942538313 ||AX-Z||: 31096 ||Xk+1-Xk|| 38451.34084430783 sw: 15
iter#: 4 F(Z): 1283.8261168089757 f(z): 1177.7481956092654 ||AX-Z||: 28608 ||Xk+1-Xk|| 38119.11656408474 sw: 15
iter#: 5 F(Z): 1306.3706137087347 f(z): 1195.6391511447077 ||AX-Z||: 26120 ||Xk+1-Xk|| 38616.58536280104 sw: 15
iter#: 6 F(Z): 1286.4732247177924 f(z): 1180.2810057805718 ||AX-Z||: 23632 ||Xk+1-Xk|| 38094.622941209615 sw: 15
iter#: 7 F(Z): 1289.034290417763 f(z): 1191.5072850703339 ||AX-Z||: 21144 ||Xk+1-Xk|| 38781.83636663251 sw: 15
iter#: 8 F(Z): 1292.7495000953584 f(z): 1177.7928614485832 ||AX-Z||: 18656 ||Xk+1-Xk|| 37928.80831609284 sw: 15
iter#: 9 F(Z): 1272.1747465514807 f(z): 1187.3376121093384 ||AX-Z||: 16168 ||Xk+1-Xk|| 38947.09561891241 sw: 15
iter#: 0 F(Z): 2163.217040964759 f(z): 1197.6980128607704 ||AX-Z||: 13680 ||Xk+1-Xk|| 38987.08141648534 sw: 8
iter#: 1 F(Z): 2162.8449109053986 f(z): 1210.5459201024096 ||AX-Z||: 13678 ||Xk+1-Xk|| 39123.45641818397 sw: 8
iter#: 2 F(Z): 2149.1812892796274 f(z): 1197.5269427608052 ||AX-Z||: 13676 ||Xk+1-Xk|| 39104.46149463173 sw: 8
iter#: 3 F(Z): 2171.073321433095 f(z): 1210.4593120321904 ||AX-Z||: 13674 ||Xk+1-Xk|| 39053.96290487171 sw: 8
iter#: 4 F(Z): 2131.2224864932537 f(z): 1197.223278376764 ||AX-Z||: 13676 ||Xk+1-Xk|| 39256.49242161948 sw: 8
...
```
## visualization
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
