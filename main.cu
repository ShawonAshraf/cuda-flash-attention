#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from GPU!\n");
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
} 
