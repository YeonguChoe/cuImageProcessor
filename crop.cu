#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitmap.cuh"
#include "grayscale.cuh"

__global__ void crop(PixelData *original_image, int original_pitch, int start_i, int start_j, int crop_width, int crop_height,
                     PixelData *cropped_image, int cropped_pitch)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < crop_width && j < crop_height)
    {
        PixelData *source = original_image + ((start_j + j) * original_pitch) + (start_i + i);
        PixelData *destination = cropped_image + j * cropped_pitch + i;
        *destination = *source;
    }
}

__host__ bool crop(char *filename, int start_x, int start_y, int end_x, int end_y)
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
    PixelData cpu_image[height][width];
    fseek(file, bitmapHeader.bitmapFileHeader.offset, SEEK_SET);
    for (int h = 0; h < height; ++h)
    {
        fread(&cpu_image[h], sizeof(PixelData), width, file);
        fseek(file, padding, SEEK_CUR);
    }
    fclose(file);

    // GPU image
    // original image
    PixelData *gpu_image;
    size_t pitch; // row including padding
    cudaMallocPitch(&gpu_image, &pitch, width * sizeof(PixelData), height);
    cudaMemcpy2D(gpu_image, pitch,
                 cpu_image, width * sizeof(PixelData),
                 width * sizeof(PixelData), height,
                 cudaMemcpyHostToDevice);

    // cropped image
    int crop_width = end_x - start_x, crop_height = end_y - start_y;
    PixelData *gpu_image_cropped;
    size_t pitch_cropped;
    cudaMallocPitch(&gpu_image_cropped, &pitch_cropped, crop_width * sizeof(PixelData), crop_height);

    // run kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((crop_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (crop_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    crop<<<numBlocks, threadsPerBlock>>>(gpu_image, pitch, start_x, start_y, crop_width, crop_height, gpu_image_cropped, pitch_cropped);

    // GPU -> CPU
    PixelData *cropped_cpu_image = (PixelData *)malloc(crop_width * crop_height * sizeof(PixelData));
    cudaMemcpy2D(cropped_cpu_image, crop_width * sizeof(PixelData),
                 gpu_image_cropped, pitch_cropped,
                 crop_width * sizeof(PixelData), crop_height,
                 cudaMemcpyDeviceToHost);

    // save to file
    const char *prefix = "cropped_";
    size_t filename_length = strlen(filename);
    size_t prefix_length = strlen(prefix);
    char *cropped_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(cropped_filename, prefix);
    strcat(cropped_filename, filename);

    // new image padding
    int new_padding = (4 - crop_width * channels % 4) % 4;

    // Update bitmapHeader
    bitmapHeader.bitmapInfoHeader.width = crop_width;
    bitmapHeader.bitmapInfoHeader.height = crop_height;
    bitmapHeader.bitmapInfoHeader.sizeImage = (crop_width * channels + new_padding) * crop_height;
    bitmapHeader.bitmapFileHeader.size = bitmapHeader.bitmapFileHeader.offset + (crop_width * channels + new_padding) * crop_height;

    FILE *cropped_file = fopen(cropped_filename, "wb");
    fwrite(&bitmapHeader, sizeof(BitmapHeader), 1, cropped_file);

    for (int h = 0; h < crop_height; ++h)
    {
        fwrite(cropped_cpu_image + (crop_width * h), sizeof(PixelData), crop_width, cropped_file);
        for (int p = 0; p < new_padding; ++p)
        {
            fputc(0, cropped_file);
        }
    }

    fclose(cropped_file);

    // free memory
    cudaFree(gpu_image);
    cudaFree(gpu_image_cropped);
    free(cropped_cpu_image);
    free(cropped_filename);

    return true;
}