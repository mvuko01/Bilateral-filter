#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>


#define THREADS_PER_BLOCK 16

using namespace cv;
using namespace std::chrono;

__device__ float distance(int x, int y, int i, int j) {
    return sqrtf(powf(x - i, 2) + powf(y - j, 2));
}

__device__ double gaussian(float x, double sigma) {
    return exp(-(powf(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

__global__ void applyBilateralFilterCUDA(const unsigned char* source, unsigned char* filteredImage, int width, int height, int diameter, double sigmaI, double sigmaS) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    extern __shared__ unsigned char sharedSource[];

    // Shared memory dimensions
    int sharedWidth = blockDim.x + 2 * diameter/2;
    int sharedHeight = blockDim.y + 2 * diameter/2;

    // Shared memory index
    int sharedIndex = (threadIdx.y + diameter / 2) * sharedWidth + (threadIdx.x + diameter / 2);
    int imageIndex = y * width + x;

    // Copy data from global memory to shared memory, including padding
    for (int channel = 0; channel < 3; channel++) {
        // Central shared memory
        sharedSource[sharedIndex + channel * sharedWidth * sharedHeight] = source[imageIndex * 3 + channel];

        // Top and bottom padding
        if (threadIdx.y < diameter / 2) {
            int topSharedIndex = threadIdx.y * sharedWidth + threadIdx.x + diameter / 2;
            int topImageIndex = (y - diameter / 2) * width + x;

            sharedSource[topSharedIndex + channel * sharedWidth * sharedHeight] = source[topImageIndex * 3 + channel];

            int bottomSharedIndex = (threadIdx.y + blockDim.y + diameter / 2) * sharedWidth + threadIdx.x + diameter / 2;
            int bottomImageIndex = (y + blockDim.y) * width + x;

            sharedSource[bottomSharedIndex + channel * sharedWidth * sharedHeight] = source[bottomImageIndex * 3 + channel];
        }

        // Left and right padding
        if (threadIdx.x < diameter / 2) {
            int leftSharedIndex = (threadIdx.y + diameter / 2) * sharedWidth + threadIdx.x;
            int leftImageIndex = y * width + (x - diameter / 2);

            sharedSource[leftSharedIndex + channel * sharedWidth * sharedHeight] = source[leftImageIndex * 3 + channel];

            int rightSharedIndex = (threadIdx.y + diameter / 2) * sharedWidth + threadIdx.x + blockDim.x + diameter / 2;
            int rightImageIndex = y * width + (x + blockDim.x);

            sharedSource[rightSharedIndex + channel * sharedWidth * sharedHeight] = source[rightImageIndex * 3 + channel];
        }

    }
    __syncthreads();

    for (int channel = 0; channel < 3; channel++) {
        double iFiltered = 0;
        double wP = 0;
        int neighbor_x = 0;
        int neighbor_y = 0;
        int half = diameter / 2;
        double gi = 0;
        double gs = 0;
        double w = 0;

        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
                neighbor_x = x - (half - i);
                neighbor_y = y - (half - j);

                if (neighbor_x >= width) {
                    //neighbor_x -= width;
                    neighbor_x = width - 1;
                }
                else if (neighbor_x < 0) {
                    neighbor_x = 0;
                }

                if (neighbor_y >= height) {
                    //neighbor_y -= height;
                    neighbor_y = height - 1;
                }
                else if (neighbor_y < 0) {
                    neighbor_y = 0;
                }

                gi = gaussian(sharedSource[(threadIdx.y + j) * sharedWidth + (threadIdx.x + i) + channel * sharedWidth * sharedHeight] - sharedSource[sharedIndex + channel * sharedWidth * sharedHeight], sigmaI);
                gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
                w = gi * gs;

                iFiltered += sharedSource[(threadIdx.y + j) * sharedWidth + (threadIdx.x + i) + channel * sharedWidth * sharedHeight] * w;
                wP += w;
            }
        }

        iFiltered /= wP;
        filteredImage[imageIndex * 3 + channel] = static_cast<unsigned char>(iFiltered);
    }
}

void bilateralFilterCUDA(const cv::Mat& source, cv::Mat& filteredImage, int diameter, double sigmaI, double sigmaS) {
    int width = source.cols;
    int height = source.rows;

    unsigned char* d_source;
    unsigned char* d_filteredImage;

    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    cudaMalloc((void**)&d_source, imageSize);
    cudaMalloc((void**)&d_filteredImage, imageSize);

    cudaMemcpy(d_source, source.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    //Calculate the shared memory size
    int sharedMemorySize = (blockSize.x + diameter - 1) * (blockSize.y + diameter - 1) * 3 * sizeof(unsigned char);

    applyBilateralFilterCUDA <<< gridSize, blockSize, sharedMemorySize >> > (d_source, d_filteredImage, width, height, diameter, sigmaI, sigmaS);

    cudaMemcpy(filteredImage.data, d_filteredImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_source);
    cudaFree(d_filteredImage);
}

int main(int argc, char** argv) {
    Mat src;
    src = imread("C:/Users/Korisnik/Desktop/NAR/noise.jpg");

    if (!src.data) {
        std::cout << "No image data" << std::endl;
        return -1;
    }
    auto start = high_resolution_clock::now();

    Mat filteredImage(src.size(), src.type());

    bilateralFilterCUDA(src, filteredImage, 8, 20.0, 20.0);

    imwrite("C:/Users/Korisnik/Desktop/NAR/filtered_image_own_CUDA_shared.jpg", filteredImage);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time of execution on GPU using shared memory is: " << duration.count() << " ms" << std::endl;
    return 0;
}
