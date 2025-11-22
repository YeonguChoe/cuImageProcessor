#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>

#pragma pack(push, 1)
typedef struct
{
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BitmapFileHeader;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression;
    uint32_t sizeImage;
    int32_t xPelsPerMeter;
    int32_t yPelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
} BitmapInfoHeader;
#pragma pack(pop)

typedef struct
{
    BitmapFileHeader bitmapFileHeader;
    BitmapInfoHeader bitmapInfoHeader;
} BitmapHeader;

typedef struct
{
    uint8_t blue;
    uint8_t green;
    uint8_t red;
} PixelData;

#endif