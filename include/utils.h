#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Utility function to check for CUDA errors
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Function to print a tensor (for debugging)
template<typename T>
void print_tensor(T* tensor, int batch_size, int seq_len, int head_dim, const std::string& name);
