// Copyright 2025 Yeongu Choe

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitmap.cuh"
#include "grayscale.cuh"

__global__ void grayscale(PixelData *devPtr, size_t pitch, int width, int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= width || j >= height)
    {
        return;
    }

    PixelData *pixelPtr = (PixelData *)((char *)devPtr + j * pitch) + i;

    uint8_t gray = (uint8_t)(0.299f * pixelPtr->red + 0.587f * pixelPtr->green + 0.114f * pixelPtr->blue);

    pixelPtr->red = gray;
    pixelPtr->green = gray;
    pixelPtr->blue = gray;
}

__host__ bool grayscale(const char *filename)
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
    // allocate memory
    PixelData cpu_image[height][width];

    // copy image data to cpu_image
    fseek(file, bitmapHeader.bitmapFileHeader.offset, SEEK_SET);

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            fread(&cpu_image[h][w], sizeof(PixelData), 1, file);
        }
        fseek(file, padding, SEEK_CUR);
    }

    fclose(file);

    // GPU image
    PixelData *gpu_image;
    size_t pitch;
    // pitch is calculated automatically
    cudaMallocPitch(&gpu_image, &pitch, width * sizeof(PixelData), height);
    cudaMemcpy2D(gpu_image, pitch,
                 cpu_image, width * sizeof(PixelData),
                 width * sizeof(PixelData), height,
                 cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    grayscale<<<numBlocks, threadsPerBlock>>>(gpu_image, pitch, width, height);

    // GPU -> CPU
    PixelData *grayscaled_cpu_image = (PixelData *)malloc(width * height * sizeof(PixelData));
    cudaMemcpy2D(grayscaled_cpu_image, width * sizeof(PixelData),
                 gpu_image, pitch,
                 width * sizeof(PixelData), height,
                 cudaMemcpyDeviceToHost);

    // create filename
    const char *prefix = "grayscaled_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *grayscaled_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(grayscaled_filename, prefix);
    strcat(grayscaled_filename, filename);

    // save to file
    FILE *grayscaled_file = fopen(grayscaled_filename, "wb");
    fwrite(&bitmapHeader, sizeof(BitmapHeader), 1, grayscaled_file);

    for (int h = 0; h < height; ++h)
    {
        fwrite(grayscaled_cpu_image + (width * h), sizeof(PixelData), width, grayscaled_file);
        for (int p = 0; p < padding; ++p)
        {
            fputc(0, grayscaled_file);
        }
    }

    fclose(grayscaled_file);

    // free memory
    free(grayscaled_cpu_image);
    free(grayscaled_filename);
    cudaFree(gpu_image);

    return true;
}