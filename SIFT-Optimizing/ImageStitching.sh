#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# OpenCV installation path
OPENCV_DIR="/work1/ntucourse/ntustudent003/opencv"

# Ensure the build directory exists
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Navigate to the build directory
cd build

# Run CMake with the specified compiler and OpenCV path
echo "Running CMake..."
cmake -DCMAKE_CXX_COMPILER=hipcc -DOpenCV_DIR="${OPENCV_DIR}/lib64/cmake/opencv4" ..

# Compile the project
echo "Building the project..."
make -j$(nproc)  # Use all available CPU cores for faster build

# Return to the base directory
cd ..

# Ensure the bin directory exists
if [ ! -d "bin" ]; then
    echo "Error: bin directory does not exist. Please check your build configuration."
    exit 1
fi

# Navigate to the bin directory
cd bin

# Execute the program using srun
echo "Executing the program..."
# srun -p mi2104x -t 00:10:00 ./image_stitching  ../bin/overlaid_result.jpg ../imgs/ImageSrc/4-4.jpg
srun -p mi2104x -t 00:10:00 ./image_stitching  ../imgs/ImageSrc2/4-1.jpg ../imgs/ImageSrc2/4-2.jpg > image_stitching_profile.txt
echo "Execution completed successfully."
