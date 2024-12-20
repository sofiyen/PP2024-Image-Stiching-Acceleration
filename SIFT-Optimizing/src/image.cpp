#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <hip/hip_runtime.h>
#include "image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void checkHIPError(hipError_t error) {
    if (error != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(error));
        exit(-1);
    }
}

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    data = new float[size];
#pragma omp parallel for collapse(3)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4)
        channels = 3; //ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}

Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        //std::cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
__global__ void transformImageDataKernel(
    const float *__restrict__ input,
    unsigned char *__restrict__ output,
    int width,
    int height,
    int channels)
{
    // Calculate global thread position
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int pixel_offset = y * width + x;

    for (int c = 0; c < channels; c++)
    {
        // Input is channel-major: c * (height * width) + y * width + x
        const int src_idx = c * (height * width) + pixel_offset;
        // Output is pixel-major: y * (width * channels) + x * channels + c
        const int dst_idx = pixel_offset * channels + c;

        output[dst_idx] = static_cast<unsigned char>(min(max(input[src_idx] * 255.0f, 0.0f), 255.0f));
    }
}

bool Image::save(std::string file_path)
{
    // Calculate optimal thread block dimensions
    constexpr int BLOCK_SIZE_X = 16;
    constexpr int BLOCK_SIZE_Y = 16;

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim(
        (width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Allocate device memory
    float *d_input = nullptr;
    unsigned char *d_output = nullptr;

    hipMalloc(&d_input, width * height * channels * sizeof(float));
    hipMalloc(&d_output, width * height * channels * sizeof(unsigned char));

    // Copy input data to device
    hipMemcpy(d_input, data, width * height * channels * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel
    hipLaunchKernelGGL(transformImageDataKernel,
                       gridDim,
                       blockDim,
                       0, 0,
                       d_input,
                       d_output,
                       width,
                       height,
                       channels);

    // Allocate host memory for output
    unsigned char *out_data = new unsigned char[width * height * channels];

    // Copy result back to host
    hipMemcpy(out_data, d_output, width * height * channels * sizeof(unsigned char), hipMemcpyDeviceToHost);

    // Clean up device memory
    hipFree(d_input);
    hipFree(d_output);

    // Save image using stbi
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
    {
        std::cerr << "Failed to save image: " << file_path << "\n";
    }

    delete[] out_data;
    return success;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= height)
        y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int size = width * height * channels;
    // #pragma unroll 4
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

//map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    float value = 0;
    // # pragma omp parallel for
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

__global__ void verticalConvolution(const float *input, float *output,
                                    const float *kernel, int width, int height,
                                    int kernelSize)
{
    extern __shared__ float s_kernel[]; // Dynamic shared memory allocation

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Load kernel into shared memory
    for (int i = threadIdx.x + threadIdx.y * blockDim.x;
         i < kernelSize;
         i += blockDim.x * blockDim.y)
    {
        if (i < kernelSize)
        {
            s_kernel[i] = kernel[i];
        }
    }
    __syncthreads();

    float sum = 0.0f;
    int center = kernelSize / 2;

    // Compute convolution
#pragma unroll 4
    for (int k = 0; k < kernelSize; k++)
    {
        int dy = y + k - center;
        dy = max(0, min(dy, height - 1));
        sum += input[dy * width + x] * s_kernel[k];
    }

    output[y * width + x] = sum;
}

__global__ void horizontalConvolution(const float *input, float *output,
                                      const float *kernel, int width, int height,
                                      int kernelSize)
{
    extern __shared__ float s_kernel[]; // Dynamic shared memory allocation

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Load kernel into shared memory
    for (int i = threadIdx.x + threadIdx.y * blockDim.x;
         i < kernelSize;
         i += blockDim.x * blockDim.y)
    {
        if (i < kernelSize)
        {
            s_kernel[i] = kernel[i];
        }
    }
    __syncthreads();

    float sum = 0.0f;
    int center = kernelSize / 2;

    // Compute convolution
#pragma unroll 4
    for (int k = 0; k < kernelSize; k++)
    {
        int dx = x + k - center;
        dx = max(0, min(dx, width - 1));
        sum += input[y * width + dx] * s_kernel[k];
    }

    output[y * width + x] = sum;
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma)
{
    auto start_compute = std::chrono::high_resolution_clock::now();
    assert(img.channels == 1);

    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)  size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++)
        kernel.data[k] /= sum;
    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    float *d_input, *d_temp, *d_output, *d_kernel;
    size_t imageSize = img.width * img.height * sizeof(float);
    size_t kernelSize = size * sizeof(float);

    checkHIPError(hipMalloc(&d_input, imageSize));
    checkHIPError(hipMalloc(&d_temp, imageSize));
    checkHIPError(hipMalloc(&d_output, imageSize));
    checkHIPError(hipMalloc(&d_kernel, kernelSize));

    checkHIPError(hipMemcpy(d_input, img.data, imageSize, hipMemcpyHostToDevice));
    checkHIPError(hipMemcpy(d_kernel, kernel.data, kernelSize, hipMemcpyHostToDevice));

    dim3 blockSize2D(32, 8);
    dim3 gridSize2D(
        (img.width + blockSize2D.x - 1) / blockSize2D.x,
        (img.height + blockSize2D.y - 1) / blockSize2D.y
    );
    size_t sharedMemSize = size * sizeof(float);
    verticalConvolution<<<gridSize2D, blockSize2D, sharedMemSize>>>(
        d_input, d_temp, d_kernel, img.width, img.height, size);
    horizontalConvolution<<<gridSize2D, blockSize2D, sharedMemSize>>>(
        d_temp, d_output, d_kernel, img.width, img.height, size);
    checkHIPError(hipDeviceSynchronize());
    checkHIPError(hipMemcpy(filtered.data, d_output, imageSize, hipMemcpyDeviceToHost));
    
    checkHIPError(hipFree(d_input));
    checkHIPError(hipFree(d_temp));
    checkHIPError(hipFree(d_output));
    checkHIPError(hipFree(d_kernel));
    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute);
    return filtered;
}

void draw_point(Image& img, int x, int y, int size)
{

    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}

