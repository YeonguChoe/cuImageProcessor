#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>

__global__ void cuda_grayscale(unsigned char *image, int width, int height, int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        int idx = (j * width + i) * channels;
        unsigned char red = image[idx];
        unsigned char green = image[idx + 1];
        unsigned char blue = image[idx + 2];

        unsigned char gray = static_cast<unsigned char>(0.299f * red + 0.587 * green + 0.114f * blue);

        // overwrite
        image[idx] = gray;
    }
}

bool grayscale(const char *filename)
{
    int width, height, channels;
    unsigned char *cpu_image = stbi_load(filename, &width, &height, &channels, 0);

    // allocate image to gpu
    unsigned char *gpu_image = nullptr;
    cudaMalloc(&gpu_image, width * height * channels);
    cudaMemcpy(gpu_image, cpu_image, width * height * channels, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // run kernel
    cuda_grayscale<<<numBlocks, threadsPerBlock>>>(gpu_image, width, height, 3);

    unsigned char *result_image = new unsigned char[width * height];
    cudaMemcpy(result_image, gpu_image, width * height * 1, cudaMemcpyDeviceToHost);

    // save image
    char *prefix = "grayscale_";
    size_t prefix_length = strlen(prefix);
    size_t filename_length = strlen(filename);
    char *grayscale_filename = (char *)malloc(prefix_length + filename_length + 1);
    strcpy(grayscale_filename, prefix);
    strcat(grayscale_filename, filename);
    stbi_write_png(grayscale_filename, width, height, 1, result_image, width * 1);

    // free memory
    delete[] result_image;
    cudaFree(gpu_image);
    stbi_image_free(cpu_image);
    free(grayscale_filename);

    return true;
}