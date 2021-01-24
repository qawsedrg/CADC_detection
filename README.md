## Features
- CADC 对地侦查打击比赛程序识别部分
- 20 FPS on TX2
- multi-threading
- TensorRT accelerated
## Dependencies
- [tkDNN](https://github.com/ceccocats/tkDNN)
- [darknet](https://github.com/AlexeyAB/darknet)
- [libtorch](https://github.com/pytorch/pytorch)
- [TensorRT](https://github.com/NVIDIA/TensorRT)
- CUDA
- cuDNN
- [OpenCV](https://github.com/opencv/opencv)

## Build
- Build darknet by allowing OpenCV, GPU and cuDNN
- add the following code to public method of class `DetectionNN` in `tkDNN-master/include/tkDNN/DetectionNN.h`
```C++
void setthreshold(float threshold) confThreshold=threshold;
```
- rebuild libtorch by setting to allow `CXX11_ABI` in `TorchConfig.cmake`
```C++
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
endif()
```