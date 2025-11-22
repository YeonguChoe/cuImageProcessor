#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "resize.cuh"
#include "bitmap.cuh"

// kernel runs once, processing the image simultaneously
// each thread process each pixel in original image
__global__ void resize(PixelData *original_image, int original_width, int original_height,
                       PixelData *resized_image, int width, int height, int channels)
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

bool resize(const char *filename, int width, int height)
{

    // Read BMP file
    FILE *file = fopen(filename, "rb");

    BitmapHeader bitmapHeader;
    fread(&bitmapHeader, sizeof(BitmapHeader), 1, file);

    int original_width = bitmapHeader.bitmapInfoHeader.width;
    int original_height = bitmapHeader.bitmapInfoHeader.height;
    int channels = bitmapHeader.bitmapInfoHeader.bitCount / 8;
    int original_padding = (4 - original_width * channels % 4) % 4;

    // CPU image
    PixelData *cpu_image = (PixelData *)malloc(original_width * original_height * sizeof(PixelData));
    fseek(file, bitmapHeader.bitmapFileHeader.offset, SEEK_SET);

    for (int h = 0; h < original_height; ++h)
    {
        fread(cpu_image + (original_width * h), sizeof(PixelData), original_width, file);
        fseek(file, original_padding, SEEK_CUR);
    }

    fclose(file);

    // GPU image
    PixelData *gpu_image = nullptr;
    cudaMalloc(&gpu_image, original_width * original_height * sizeof(PixelData));
    cudaMemcpy(gpu_image, cpu_image, original_width * original_height * sizeof(PixelData), cudaMemcpyHostToDevice);

    // resized GPU image
    PixelData *resized_gpu_image = nullptr;
    cudaMalloc(&resized_gpu_image, width * height * sizeof(PixelData));

    // kernel setup
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // run kernel
    resize<<<numBlocks, threadsPerBlock>>>(gpu_image, original_width, original_height,
                                           resized_gpu_image, width, height, channels);

    // GPU -> CPU
    PixelData *resized_cpu_image = (PixelData *)malloc(width * height * sizeof(PixelData));
    cudaMemcpy(resized_cpu_image, resized_gpu_image, width * height * sizeof(PixelData), cudaMemcpyDeviceToHost);

    // create filename
    const char *prefix = "resized_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *resized_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(resized_filename, prefix);
    strcat(resized_filename, filename);

    // new image padding
    int padding = (4 - width * channels % 4) % 4;

    // Update bitmapHeader
    bitmapHeader.bitmapInfoHeader.width = width;
    bitmapHeader.bitmapInfoHeader.height = height;
    bitmapHeader.bitmapInfoHeader.sizeImage = (width * channels + padding) * height;
    bitmapHeader.bitmapFileHeader.size = bitmapHeader.bitmapFileHeader.offset + (width * channels + padding) * height;

    // save to file
    FILE *resized_file = fopen(resized_filename, "wb");
    fwrite(&bitmapHeader, sizeof(BitmapHeader), 1, resized_file);

    for (int h = 0; h < height; ++h)
    {
        fwrite(resized_cpu_image + (width * h), sizeof(PixelData), width, resized_file);
        for (int p = 0; p < padding; ++p)
        {
            fputc(0, resized_file);
        }
    }

    fclose(resized_file);

    // free memory
    free(cpu_image);
    cudaFree(gpu_image);
    cudaFree(resized_gpu_image);
    free(resized_cpu_image);
    free(resized_filename);

    return true;
}