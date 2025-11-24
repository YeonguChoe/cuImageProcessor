#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitmap.cuh"
#include "crop.cuh"

__global__ void crop(cudaPitchedPtr gpu_original_image, int original_width, int original_height, int original_depth,
                     cudaPitchedPtr gpu_cropped_image, int crop_width, int crop_height, int crop_depth,
                     int start_x, int start_y)
{
    // coordinate at cropped image
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= crop_width || j >= crop_height || k >= crop_depth)
    {
        return;
    }

    // coordinate at original image
    int source_i = i + start_x;
    int source_j = j + start_y;
    int source_k = k;

    uint8_t *source_slice = (uint8_t *)gpu_original_image.ptr + source_k * gpu_original_image.pitch * original_height;
    uint8_t *destination_slice = (uint8_t *)gpu_cropped_image.ptr + k * gpu_cropped_image.pitch * crop_height;

    uint8_t *source_row = source_slice + source_j * gpu_original_image.pitch;
    uint8_t *destination_row = destination_slice + j * gpu_cropped_image.pitch;

    destination_row[i] = source_row[source_i];
}

__host__ bool crop(char *filename, int start_x, int start_y, int end_x, int end_y)
{
    // read BMP file
    FILE *file = fopen(filename, "rb");

    BitmapHeader bitmapHeader;
    fread(&bitmapHeader, sizeof(BitmapHeader), 1, file);

    int original_width = bitmapHeader.bitmapInfoHeader.width;
    int original_height = bitmapHeader.bitmapInfoHeader.height;
    int channels = bitmapHeader.bitmapInfoHeader.bitCount / 8;
    int original_padding = (4 - original_width * channels % 4) % 4;

    // crop detail (exclude end value)
    int crop_width = end_x - start_x;
    int crop_height = end_y - start_y;

    // CPU image
    uint8_t cpu_image[channels][original_height][original_width]; // channel[0]: blue, channel[1]: green, channel[2]: red
    fseek(file, bitmapHeader.bitmapFileHeader.offset, SEEK_SET);

    PixelData *original_image_row_buffer = (PixelData *)malloc(original_width * sizeof(PixelData));
    for (int h = 0; h < original_height; ++h)
    {
        fread(original_image_row_buffer, sizeof(PixelData), original_width, file);
        for (int w = 0; w < original_width; ++w)
        {
            cpu_image[0][h][w] = original_image_row_buffer[w].blue;
            cpu_image[1][h][w] = original_image_row_buffer[w].green;
            cpu_image[2][h][w] = original_image_row_buffer[w].red;
        }
        fseek(file, original_padding, SEEK_CUR);
    }

    fclose(file);

    // GPU image
    // GPU original image
    cudaExtent gpu_original_image_extent = make_cudaExtent(original_width, original_height, channels);
    cudaPitchedPtr gpu_original_image;
    cudaMalloc3D(&gpu_original_image, gpu_original_image_extent);

    cudaMemcpy3DParms gpu_original_image_params = {0};
    gpu_original_image_params.srcPtr = make_cudaPitchedPtr(
        cpu_image,
        original_width * sizeof(uint8_t),
        original_width,
        original_height);
    gpu_original_image_params.dstPtr = gpu_original_image;
    gpu_original_image_params.extent = gpu_original_image_extent;
    gpu_original_image_params.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&gpu_original_image_params);

    // GPU cropped image
    cudaExtent gpu_cropped_image_extent = make_cudaExtent(crop_width, crop_height, channels);
    cudaPitchedPtr gpu_cropped_image;
    cudaMalloc3D(&gpu_cropped_image, gpu_cropped_image_extent);

    cudaMemcpy3DParms gpu_cropped_image_params = {0};
    gpu_cropped_image_params.srcPtr = make_cudaPitchedPtr(
        cpu_image,
        crop_width * sizeof(uint8_t),
        crop_width,
        crop_height);
    gpu_cropped_image_params.dstPtr = gpu_cropped_image;
    gpu_cropped_image_params.extent = gpu_cropped_image_extent;
    gpu_cropped_image_params.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&gpu_cropped_image_params);

    return true;
}