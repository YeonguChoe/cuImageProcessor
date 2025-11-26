#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitmap.cuh"
#include "flip.cuh"

__global__ void flip_horizontal(PixelData *gpu_image, size_t pitch, int width, int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // set range
    if (i > width / 2 || j >= height)
    {
        return;
    }

    PixelData *row = (PixelData *)((char *)gpu_image + pitch * j);
    PixelData temp = *(row + i);
    *(row + i) = *(row + width - 1 - i);
    *(row + width - 1 - i) = temp;
}

__host__ bool flip_horizontal(char *filename)
{
    // Read BMP file
    FILE *file = fopen(filename, "rb");

    BitmapHeader bitmapHeader;
    fread(&bitmapHeader, sizeof(BitmapHeader), 1, file);

    int width = bitmapHeader.bitmapInfoHeader.width;
    int height = bitmapHeader.bitmapInfoHeader.height;
    int channels = bitmapHeader.bitmapInfoHeader.bitCount / 8;
    int padding = (4 - width * channels % 4) % 4;

    // CPU image
    PixelData *cpu_image = (PixelData *)malloc(height * width * sizeof(PixelData));
    for (int h = 0; h < height; ++h)
    {
        fread(cpu_image + (width * h), sizeof(PixelData), width, file);
        fseek(file, padding, SEEK_CUR);
    }

    fclose(file);

    // GPU image
    PixelData *gpu_image;
    size_t pitch;
    cudaMallocPitch(&gpu_image, &pitch, width * sizeof(PixelData), height);
    cudaMemcpy2D(gpu_image, pitch,
                 cpu_image, width * sizeof(PixelData),
                 width * sizeof(PixelData), height,
                 cudaMemcpyHostToDevice);

    // run kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    flip_horizontal<<<numBlocks, threadsPerBlock>>>(gpu_image, pitch, width, height);

    // GPU -> CPU
    PixelData *flipped_cpu_image = (PixelData *)malloc(width * height * sizeof(PixelData));
    cudaMemcpy2D(flipped_cpu_image, width * sizeof(PixelData),
                 gpu_image, pitch,
                 width * sizeof(PixelData), height,
                 cudaMemcpyDeviceToHost);

    // save to file
    const char *prefix = "horizontally_flipped_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *horizontally_flipped_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(horizontally_flipped_filename, prefix);
    strcat(horizontally_flipped_filename, filename);

    // save to file
    FILE *horizontally_flipped_file = fopen(horizontally_flipped_filename, "wb");
    fwrite(&bitmapHeader, sizeof(BitmapHeader), 1, horizontally_flipped_file);

    for (int h = 0; h < height; ++h)
    {
        fwrite(flipped_cpu_image + (width * h), sizeof(PixelData), width, horizontally_flipped_file);
        for (int p = 0; p < padding; ++p)
        {
            fputc(0, horizontally_flipped_file);
        }
    }

    fclose(horizontally_flipped_file);

    // free memory
    free(cpu_image);
    free(flipped_cpu_image);
    free(horizontally_flipped_filename);
    cudaFree(gpu_image);

    return true;
}

__global__ void flip_vertical(PixelData *gpu_image, size_t pitch, int width, int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // set range

    if (i >= width || j >= height / 2)
    {
        return;
    }

    PixelData *top_row = (PixelData *)((char *)gpu_image + pitch * j);
    PixelData *bottom_row = (PixelData *)((char *)gpu_image + pitch * (height - 1 - j));

    PixelData temp = *(top_row + i);
    *(top_row + i) = *(bottom_row + i);
    *(bottom_row + i) = temp;
}

__host__ bool flip_vertical(char *filename)
{
    // Read BMP file
    FILE *file = fopen(filename, "rb");

    BitmapHeader bitmapHeader;
    fread(&bitmapHeader, sizeof(BitmapHeader), 1, file);

    int width = bitmapHeader.bitmapInfoHeader.width;
    int height = bitmapHeader.bitmapInfoHeader.height;
    int channels = bitmapHeader.bitmapInfoHeader.bitCount / 8;
    int padding = (4 - ((width * channels) % 4)) % 4;

    // CPU image
    PixelData *cpu_image = (PixelData *)malloc(height * width * sizeof(PixelData));
    for (int h = 0; h < height; ++h)
    {
        fread(cpu_image + (width * h), sizeof(PixelData), width, file);
        fseek(file, padding, SEEK_CUR);
    }

    fclose(file);

    // GPU image
    PixelData *gpu_image;
    size_t pitch;
    cudaMallocPitch(&gpu_image, &pitch, width * sizeof(PixelData), height);
    cudaMemcpy2D(gpu_image, pitch,
                 cpu_image, width * sizeof(PixelData),
                 width * sizeof(PixelData), height,
                 cudaMemcpyHostToDevice);

    // run kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    flip_vertical<<<numBlocks, threadsPerBlock>>>(gpu_image, pitch, width, height);

    // GPU -> CPU
    PixelData *flipped_cpu_image = (PixelData *)malloc(width * height * sizeof(PixelData));
    cudaMemcpy2D(flipped_cpu_image, width * sizeof(PixelData),
                 gpu_image, pitch,
                 width * sizeof(PixelData), height,
                 cudaMemcpyDeviceToHost);

    // save to file
    const char *prefix = "vertically_flipped_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *vertically_flipped_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(vertically_flipped_filename, prefix);
    strcat(vertically_flipped_filename, filename);

    // save to file
    FILE *vertically_flipped_file = fopen(vertically_flipped_filename, "wb");
    fwrite(&bitmapHeader, sizeof(BitmapHeader), 1, vertically_flipped_file);

    for (int h = 0; h < height; ++h)
    {
        fwrite(flipped_cpu_image + (width * h), sizeof(PixelData), width, vertically_flipped_file);
        for (int p = 0; p < padding; ++p)
        {
            fputc(0, vertically_flipped_file);
        }
    }

    fclose(vertically_flipped_file);

    // free memory
    free(cpu_image);
    free(flipped_cpu_image);
    free(vertically_flipped_filename);
    cudaFree(gpu_image);

    return true;
}