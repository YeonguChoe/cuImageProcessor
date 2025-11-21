#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "bitmap.cuh"

// kernel runs once, processing the image simultaneously
// each thread process each pixel in original image
__global__ void resize(unsigned char *original_image, int original_width, int original_height,
                       unsigned char *resized_image, int width, int height, int channels)
{
    // grid size is resized image size
    // blockIdx: block's coordinate within the grid
    // blockDim (constant value): block dimension
    // threadIdx: thread's coordinate within a block
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = (blockDim.y * blockIdx.y) + threadIdx.y;

    float scale_to_original_i = (float)original_width / (float)width;
    float scale_to_original_j = (float)original_height / (float)height;

    // sampling: select pixel by sampling from original image
    int pixel_at_original_i = (int)(i * scale_to_original_i);
    int pixel_at_original_j = (int)(j * scale_to_original_j);

    for (int channel = 0; channel < channels; ++channel)
    {
        resized_image[((width * j) + i) * channels + channel] = original_image[((original_width * pixel_at_original_j) + pixel_at_original_i) * channels + channel];
    }
}

__host__ bool resize(const char *filename, int width, int height)
{

    FILE *file = fopen(filename, "rb");

    BMPFileHeader fileHeader;
    fread(&fileHeader, sizeof(BMPFileHeader), 1, file);

    

    int original_width,
        original_height, channels;
    unsigned char *cpu_image = stbi_load(filename, &original_width, &original_height, &channels, 0);

    // cpu -> gpu
    unsigned char *gpu_image = nullptr;
    cudaMalloc(&gpu_image, original_width * original_height * channels * sizeof(unsigned char));
    cudaMemcpy(gpu_image, cpu_image, original_width * original_height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // output at gpu
    unsigned char *resized_gpu_image = nullptr;
    cudaMalloc(&resized_gpu_image, width * height * channels * sizeof(unsigned char));

    // kernel setup
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // run kernel
    resize<<<numBlocks, threadsPerBlock>>>(gpu_image, original_width, original_height,
                                           resized_gpu_image, width, height, channels);

    // gpu -> cpu
    unsigned char *resized_cpu_image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    cudaMemcpy(resized_cpu_image, resized_gpu_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // save to file
    const char *prefix = "resize_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *resize_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(resize_filename, prefix);
    strcat(resize_filename, filename);
    stbi_write_png(resize_filename, width, height, channels, resized_cpu_image, width * channels);

    // free memory
    cudaFree(gpu_image);
    cudaFree(resized_gpu_image);
    stbi_image_free(cpu_image);
    free(resized_cpu_image);
    free(resize_filename);

    return true;
}