#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitmap.cuh"
#include "grayscale.cuh"

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

    

    return true;
}