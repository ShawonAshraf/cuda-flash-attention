#include <iostream>
#include <vector>
#include <cstdlib>
#include "flash_attention.h"
#include "utils.h"

int main() {
    // Define tensor dimensions
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 256;
    const int head_dim = 64;

    // Allocate memory for tensors on the GPU
    float *q, *k, *v, *output;
    const int tensor_size = batch_size * num_heads * seq_len * head_dim;
    CHECK_CUDA_ERROR(cudaMalloc(&q, tensor_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&k, tensor_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&v, tensor_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&output, tensor_size * sizeof(float)));

    // Initialize Q, K, V with random data
    std::vector<float> h_q(tensor_size);
    std::vector<float> h_k(tensor_size);
    std::vector<float> h_v(tensor_size);
    for(int i = 0; i < tensor_size; ++i) {
        h_q[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        h_k[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        h_v[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(q, h_q.data(), tensor_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(k, h_k.data(), tensor_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(v, h_v.data(), tensor_size * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Running Flash Attention..." << std::endl;

    // Call the flash attention forward pass
    flash_attention_forward(q, k, v, output, batch_size, num_heads, seq_len, head_dim);

    std::cout << "Flash Attention finished." << std::endl;
    
    // For now, let's just prove the code runs. We'll verify correctness later.
    print_tensor(output, batch_size, 1, head_dim, "Output");

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(q));
    CHECK_CUDA_ERROR(cudaFree(k));
    CHECK_CUDA_ERROR(cudaFree(v));
    CHECK_CUDA_ERROR(cudaFree(output));

    return 0;
} 
