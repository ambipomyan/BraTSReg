# BraTSReg
Prepared for MICCAI 2022 challenge

## problem located
- performance and tuning
- converge toooooo fast
- take a look at the indices of moving/fixed images in function searchMin()

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
block radius(HWC): 3 3 1
init search window radius(HWC): 15 15 1
alpha: 1.0
----------------- Kid = 1 -----------------
throwing darts...
# of list points for mls: 1233
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1041.6387308873236 f(z): 1037.0698421224952 ||AX-Z||: 534.0 ||Xk+1-Xk|| 534 sw: 15
iter#: 1 F(Z): 1041.560968812555 f(z): 1037.0698421224952 ||AX-Z||: 529.5666674883059 ||Xk+1-Xk|| 0 sw: 15
iter#: 0 F(Z): 1048.8359175771475 f(z): 1040.9894363936037 ||AX-Z||: 390.02613058919826 ||Xk+1-Xk|| 155 sw: 8
iter#: 1 F(Z): 1048.8488201349974 f(z): 1040.986688202247 ||AX-Z||: 388.90446343055055 ||Xk+1-Xk|| 2 sw: 8
iter#: 2 F(Z): 1048.8469268456101 f(z): 1040.986688202247 ||AX-Z||: 388.8438640621078 ||Xk+1-Xk|| 0 sw: 8
iter#: 0 F(Z): 1061.3966125026345 f(z): 1049.0128854513168 ||AX-Z||: 219.21421302556917 ||Xk+1-Xk|| 177 sw: 4
iter#: 1 F(Z): 1061.5335670411587 f(z): 1049.1374206542969 ||AX-Z||: 203.87880448870376 ||Xk+1-Xk|| 3 sw: 4
iter#: 2 F(Z): 1061.5312081575394 f(z): 1049.1374206542969 ||AX-Z||: 203.54499653107464 ||Xk+1-Xk|| 0 sw: 4
iter#: 0 F(Z): 1074.686029647098 f(z): 1053.6398622915149 ||AX-Z||: 131.83392145261126 ||Xk+1-Xk|| 48 sw: 2
iter#: 1 F(Z): 1073.9863509461284 f(z): 1053.6874976083636 ||AX-Z||: 119.64713400291606 ||Xk+1-Xk|| 3 sw: 2
iter#: 2 F(Z): 1073.9030111655593 f(z): 1053.6874976083636 ||AX-Z||: 118.64710755540428 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 1073.8974183476312 f(z): 1053.6874976083636 ||AX-Z||: 59.587582717619625 ||Xk+1-Xk|| 0 sw: 1
----------------- Kid = 2 -----------------
throwing darts...
# of list points for mls: 1383
voxel dim by mm (HWC): 1 1 1
iter#: 0 F(Z): 1153.523213162087 f(z): 1149.0343244588003 ||AX-Z||: 562.0 ||Xk+1-Xk|| 562 sw: 15
iter#: 1 F(Z): 1153.446552454494 f(z): 1149.0343244588003 ||AX-Z||: 557.2853920672301 ||Xk+1-Xk|| 0 sw: 15
iter#: 0 F(Z): 1160.8507494842634 f(z): 1152.5721601648256 ||AX-Z||: 417.02805885462584 ||Xk+1-Xk|| 162 sw: 8
iter#: 1 F(Z): 1160.856263046153 f(z): 1152.5543151302263 ||AX-Z||: 416.0260083794797 ||Xk+1-Xk|| 1 sw: 8
iter#: 2 F(Z): 1160.8553309971467 f(z): 1152.5543151302263 ||AX-Z||: 415.9957147551108 ||Xk+1-Xk|| 0 sw: 8
iter#: 0 F(Z): 1173.9479957437143 f(z): 1160.7076877402142 ||AX-Z||: 238.54768272571636 ||Xk+1-Xk|| 175 sw: 4
iter#: 1 F(Z): 1174.3347409656271 f(z): 1160.643613521941 ||AX-Z||: 230.3249620779543 ||Xk+1-Xk|| 1 sw: 4
iter#: 2 F(Z): 1174.321628860198 f(z): 1160.643613521941 ||AX-Z||: 230.21389043398867 ||Xk+1-Xk|| 0 sw: 4
iter#: 0 F(Z): 1189.6107145315036 f(z): 1165.3376053003594 ||AX-Z||: 149.98783177915 ||Xk+1-Xk|| 58 sw: 2
iter#: 1 F(Z): 1188.5547162285075 f(z): 1165.3376053003594 ||AX-Z||: 133.9852635133273 ||Xk+1-Xk|| 0 sw: 2
iter#: 0 F(Z): 1188.5505980420858 f(z): 1165.3376053003594 ||AX-Z||: 67.2460807071014 ||Xk+1-Xk|| 0 sw: 1
```
## visualization(slice#100)
fixed image:  
![fixed image](https://github.com/ambipomyan/BraTSReg/blob/main/fixed.jpg)  
moving image:  
![moving image](https://github.com/ambipomyan/BraTSReg/blob/main/moving.jpg)  
mask image:  
![mask image](https://github.com/ambipomyan/BraTSReg/blob/main/mask.jpg)  
