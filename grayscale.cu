#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "grayscale.cuh"
#include "bitmap.cuh"

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
    PixelData **cpu_image = (PixelData **)malloc(height * sizeof(PixelData *));
    for (int h = 0; h < height; ++h)
    {
        cpu_image[h] = (PixelData *)malloc(width * sizeof(PixelData));
    }

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
    PixelData **gpu_image = nullptr;
    size_t gpu_image_pitch;
    cudaMallocPitch(&gpu_image, &gpu_image_pitch, width * sizeof(PixelData), height);




    return true;
}