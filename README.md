# Bilateral-filter

Implementation of Bilateral Filter in C++ using CUDA for RGB images.

Used for testing speeds of CPU and GPU, there are 3 code implementations:\
a) main.cpp -> implements the filter using CPU only\
b) kernel.cu -> implementation on GPU\
c) kernel2.cu -> implementation on GPU using shared memory


*NOTE: in applyBilateralFilter function there are not a lot of lines where memory access happens so the shared memory implementation is not that faster than the regular GPU implementation
