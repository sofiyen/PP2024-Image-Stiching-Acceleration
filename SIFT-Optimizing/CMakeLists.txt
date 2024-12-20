cmake_minimum_required(VERSION 3.8)

project(sift_cpp LANGUAGES CXX)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
link_directories(/lib64)

# 如果使用 Clang，添加以下設置
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++ -L/lib64")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")


set(ROCM_PATH "/opt/rocm-6.2.1" CACHE PATH "ROCm installation path")
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})


enable_language(HIP)
find_package(HIP REQUIRED)
include_directories(${HIP_INCLUDE_DIRS})


find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OpenCV_LIBS})


add_subdirectory(src_seq)
add_subdirectory(examples_seq)