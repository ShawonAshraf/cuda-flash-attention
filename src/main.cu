#include "../include/flash_attention.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Utility function to initialize tensor with random values
void initialize_tensor(float* tensor, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        tensor[i] = dis(gen);
    }
}

// Utility function to verify results (simple check)
bool verify_results(const float* result, const float* expected, size_t size, float tolerance = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (fabsf(result[i] - expected[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << result[i] << " vs " << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Test parameters
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len_q = 128;
    const int seq_len_k = 128;
    const int head_dim = 64;
    
    std::cout << "Flash Attention Test" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Num heads: " << num_heads << std::endl;
    std::cout << "Seq len Q: " << seq_len_q << std::endl;
    std::cout << "Seq len K: " << seq_len_k << std::endl;
    std::cout << "Head dim: " << head_dim << std::endl;
    
    // Calculate tensor sizes
    size_t q_size = batch_size * num_heads * seq_len_q * head_dim;
    size_t k_size = batch_size * num_heads * seq_len_k * head_dim;
    size_t v_size = k_size;
    size_t o_size = q_size;
    
    // Allocate host memory
    std::vector<float> h_Q(q_size);
    std::vector<float> h_K(k_size);
    std::vector<float> h_V(v_size);
    std::vector<float> h_O(o_size);
    
    // Initialize input tensors
    initialize_tensor(h_Q.data(), q_size);
    initialize_tensor(h_K.data(), k_size);
    initialize_tensor(h_V.data(), v_size);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, v_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, o_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), k_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), v_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure Flash Attention
    FlashAttentionConfig config(head_dim, 64, 1.0f / sqrtf(head_dim));
    
    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute Flash Attention
    flash_attention_forward(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len_q, seq_len_k, config);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, o_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify output is not all zeros (basic sanity check)
    float sum = 0.0f;
    for (size_t i = 0; i < std::min(size_t(1000), o_size); ++i) {
        sum += fabsf(h_O[i]);
    }
    
    std::cout << "Output sum (first 1000 elements): " << sum << std::endl;
    
    if (sum > 0.0f) {
        std::cout << "SUCCESS: Flash Attention executed successfully!" << std::endl;
    } else {
        std::cout << "WARNING: Output appears to be zero" << std::endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    
    return 0;
}
