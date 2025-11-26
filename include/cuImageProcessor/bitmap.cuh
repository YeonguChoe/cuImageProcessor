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

#ifndef BITMAP_CUH
#define BITMAP_CUH

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

#pragma pack(push, 1)
typedef struct
{
    BitmapFileHeader bitmapFileHeader;
    BitmapInfoHeader bitmapInfoHeader;
} BitmapHeader;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    uint8_t blue;
    uint8_t green;
    uint8_t red;
} PixelData;
#pragma pack(pop)

#endif