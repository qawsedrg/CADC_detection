cmake_minimum_required(VERSION 3.16)
project(darknet_yolo)

set(CMAKE_CXX_STANDARD 11)
set(Torch_DIR /home/chz/libtorch/share/cmake/Torch)

find_package( OpenCV REQUIRED )
find_package( Torch REQUIRED)
find_package(CUDA REQUIRED)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_75,code=sm_75
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

include_directories(include)
include_directories(/home/chz/libtorch/include)
include_directories(/home/chz/libtorch/include/torch/csrc/api/include)
include_directories(/home/chz/Downloads/tkDNN-master/include/tkDNN)
include_directories(/home/chz/TensorRT-5.1.5.0/include)
include_directories(/home/chz/TensorRT-5.1.5.0/samples/common)
include_directories(/usr/local/cuda/include )
include_directories(${CUDNN_INCLUDE_DIR})

link_directories(lib)
link_directories(/home/chz/Downloads/pytorch-1.0.1/build/lib)
link_directories(/home/chz/opencv-3.4.10/build/lib)
link_directories(/home/chz/TensorRT-5.1.5.0/lib)
link_directories(/usr/local/cuda-10.0/lib64)

set(TORCH_CUDA_LIBRARIES
        ${CUDA_CUDA_LIB}
        ${CUDA_NVRTC_LIB}
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
        ${CUDA_LIBRARIES}
        )

add_executable(darknet_yolo yolo_detector.cpp ${SOURCE_FILES} ${SOURCES})
target_link_libraries( darknet_yolo ${OpenCV_LIBS} )
target_link_libraries( darknet_yolo libc10_cuda.so )
target_link_libraries( darknet_yolo libc10.so )
target_link_libraries( darknet_yolo libcaffe2.so )
target_link_libraries( darknet_yolo libcaffe2_detectron_ops_gpu.so )
target_link_libraries( darknet_yolo libcaffe2_observers.so )
target_link_libraries( darknet_yolo libcaffe2_gpu.so )
target_link_libraries( darknet_yolo libtorch.so )
target_link_libraries( darknet_yolo ${TORCH_CUDA_LIBRARIES} )
target_link_libraries( darknet_yolo libdark.so )
target_link_libraries( darknet_yolo tkDNN.so )