## Features
- Recognition and tracking of 2-digits target for automatic aeroplane for the Chinese Aeromodelling Design Competition (CADC)
- Detection model trained using Darknet-YOLO
- [2 digits recognition, automatic parameter searching using NNI , model built using Pytorch, trained on SVHN (The Street View House Numbers)](https://github.com/qawsedrg/CADC-SVHN-PyTorch)
- Acceleration : TensorRT (tkDNN) (FP16), C++, multi-threading
- Real-time (20-25FPS) on TX2 (edge computing platform), ~ 200FPS on RTX 2070
## Dependencies
- [tkDNN](https://github.com/ceccocats/tkDNN)
- [darknet](https://github.com/AlexeyAB/darknet)
- [libtorch](https://github.com/pytorch/pytorch)
- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [OpenCV](https://github.com/opencv/opencv)
- CUDA
- cuDNN
## Build
- Build darknet by allowing OpenCV, GPU and cuDNN
- Add the following code to public method of class `DetectionNN` in `tkDNN-master/include/tkDNN/DetectionNN.h` and build tkDNN
```C++
void setthreshold(float threshold) confThreshold=threshold;
```
- Instead of using the libtorch downloaded, rebuild Torch by setting to allow `CXX11_ABI` in `TorchConfig.cmake`
```C++
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
endif()
```
