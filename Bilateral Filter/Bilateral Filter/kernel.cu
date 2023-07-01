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

    for (int channel = 0; channel < 3; channel++) {
        double iFiltered = 0;
        double wP = 0;
        int neighbor_x = 0;
        int neighbor_y = 0;
        int half = diameter / 2;
        double gs = 0;
        double gi = 0;
        double w = 0;

        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
                neighbor_x = x - (half - i);
                neighbor_y = y - (half - j);

                if (neighbor_x >= width) {
                    neighbor_x = width - 1;
                }
                else if (neighbor_x < 0) {
                    neighbor_x = 0;
                }

                if (neighbor_y >= height) {
                    neighbor_y = height - 1;
                }
                else if (neighbor_y < 0) {
                    neighbor_y = 0;
                }

                gi = gaussian(source[(neighbor_y * width + neighbor_x) * 3 + channel] - source[(y * width + x) * 3 + channel], sigmaI);
                gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
                w = gi * gs;

                iFiltered += source[(neighbor_y * width + neighbor_x) * 3 + channel] * w;
                wP += w;
            }
        }

        iFiltered /= wP;
        filteredImage[(y * width + x) * 3 + channel] = static_cast<unsigned char>(iFiltered);
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

    applyBilateralFilterCUDA <<< gridSize, blockSize >>> (d_source, d_filteredImage, width, height, diameter, sigmaI, sigmaS);

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

    imwrite("C:/Users/Korisnik/Desktop/NAR/filtered_image_own_CUDA.jpg", filteredImage);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time of execution on GPU is: " << duration.count() << " ms" << std::endl;
    waitKey(0);

    return 0;
}






