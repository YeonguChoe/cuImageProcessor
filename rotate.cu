#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitmap.cuh"
#include "rotate.cuh"

// counterclockwise rotation

__global__ void rotate(PixelData *original_gpu_image, int original_width, int original_height, size_t original_pitch,
                       PixelData *rotated_gpu_image, size_t rotated_pitch)
{
    int rotated_i = blockDim.x * blockIdx.x + threadIdx.x;
    int rotated_j = blockDim.y * blockIdx.y + threadIdx.y;

    if (rotated_i >= original_height || rotated_j >= original_width)
    {
        return;
    }

    // rotated_image[i][j] = original_image[j][width-1-i]
    int original_i = original_width - 1 - rotated_j;
    int original_j = rotated_i;

    PixelData *source = (PixelData *)((char *)original_gpu_image + original_pitch * original_j) + original_i;
    PixelData *destination = (PixelData *)((char *)rotated_gpu_image + rotated_pitch * rotated_j) + rotated_i;

    *destination = *source;
}

__host__ bool rotate(char *filename)
{
    // Read BMP file
    FILE *file = fopen(filename, "rb");

    BitmapHeader bitmapHeader;
    fread(&bitmapHeader, sizeof(BitmapHeader), 1, file);

    int height = bitmapHeader.bitmapInfoHeader.height;
    int width = bitmapHeader.bitmapInfoHeader.width;
    int channels = bitmapHeader.bitmapInfoHeader.bitCount / 8;
    int padding = (4 - width * channels % 4) % 4;

    // CPU image
    PixelData **cpu_image = (PixelData **)malloc(height * sizeof(PixelData *));
    for (int h = 0; h < height; ++h)
    {
        *(cpu_image + h) = (PixelData *)malloc(width * sizeof(PixelData));
    }

    fseek(file, bitmapHeader.bitmapFileHeader.offset, SEEK_SET);

    for (int h = 0; h < height; ++h)
    {
        fread(*(cpu_image + h), sizeof(PixelData), width, file);
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

    // rotated image
    PixelData *rotated_gpu_image;
    size_t rotated_pitch;
    cudaMallocPitch(&rotated_gpu_image, &rotated_pitch, height * sizeof(PixelData), width);

    // run kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rotate<<<numBlocks, threadsPerBlock>>>(gpu_image, width, height, pitch, rotated_gpu_image, rotated_pitch);

    // GPU -> CPU
    PixelData *cropped_cpu_image = (PixelData *)malloc(width * height * sizeof(PixelData));
    cudaMemcpy2D(cropped_cpu_image, height * sizeof(PixelData),
                 rotated_gpu_image, rotated_pitch,
                 height, width * sizeof(PixelData),
                 cudaMemcpyDeviceToHost);

    // save to file
    const char *prefix = "rotated_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *rotated_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(rotated_filename, prefix);
    strcat(rotated_filename, filename);

    // new image padding
    int new_padding = (4 - height * channels % 4) % 4;

    // Update bitmapHeader
    bitmapHeader.bitmapInfoHeader.width = height;
    bitmapHeader.bitmapInfoHeader.height = width;
    bitmapHeader.bitmapInfoHeader.sizeImage = (height * channels + new_padding) * width;
    bitmapHeader.bitmapFileHeader.size = bitmapHeader.bitmapFileHeader.offset + (height * channels + new_padding) * width;

    FILE *rotated_file = fopen(rotated_filename, "wb");
    fwrite(&bitmapHeader, sizeof(BitmapHeader), 1, rotated_file);

    for (int w = 0; w < width; ++w)
    {
        fwrite(cropped_cpu_image + (height * w), sizeof(PixelData), height, rotated_file);
        for (int p = 0; p < new_padding; ++p)
        {
            fputc(0, rotated_file);
        }
    }

    fclose(rotated_file);

    // free memory
    for (int h = 0; h < height; ++h)
    {
        free(cpu_image[h]);
    }
    free(cpu_image);

    free(cropped_cpu_image);
    free(rotated_filename);
    cudaFree(gpu_image);
    cudaFree(rotated_gpu_image);

    return true;
}