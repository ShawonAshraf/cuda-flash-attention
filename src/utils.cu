#include "utils.h"
#include <vector>
#include <iostream>

template<typename T>
void print_tensor(T* tensor, int batch_size, int seq_len, int head_dim, const std::string& name) {
    std::cout << "Tensor: " << name << " (" << batch_size << ", " << seq_len << ", " << head_dim << ")" << std::endl;
    
    // Copy data from device to host
    std::vector<T> host_tensor(batch_size * seq_len * head_dim);
    CHECK_CUDA_ERROR(cudaMemcpy(host_tensor.data(), tensor, host_tensor.size() * sizeof(T), cudaMemcpyDeviceToHost));

    // Print the tensor data
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Batch " << b << ":" << std::endl;
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < head_dim; ++h) {
                std::cout << host_tensor[b * seq_len * head_dim + s * head_dim + h] << " ";
            }
            std::cout << std::endl;
        }
    }
}

// Explicit template instantiation for float
template void print_tensor<float>(float* tensor, int batch_size, int seq_len, int head_dim, const std::string& name);
