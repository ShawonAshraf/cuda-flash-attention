#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Flash Attention configuration
struct FlashAttentionConfig {
    int head_dim;           // Head dimension (d)
    int tile_size;          // Tile size for tiling
    float softmax_scale;    // Scaling factor for attention scores
    int block_size;         // Block size for CUDA kernels
    
    FlashAttentionConfig(int dim = 64, int tile = 64, float scale = 1.0f, int blk = 128) 
        : head_dim(dim), tile_size(tile), softmax_scale(scale), block_size(blk) {}
};

// Online softmax statistics
struct SoftmaxStats {
    float max_val;
    float sum_val;
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Function declarations
void flash_attention_forward(
    const float* Q,         // Query matrix [batch_size, num_heads, seq_len_q, head_dim]
    const float* K,         // Key matrix [batch_size, num_heads, seq_len_k, head_dim]  
    const float* V,         // Value matrix [batch_size, num_heads, seq_len_k, head_dim]
    float* O,               // Output matrix [batch_size, num_heads, seq_len_q, head_dim]
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    const FlashAttentionConfig& config = FlashAttentionConfig()
);

#endif // FLASH_ATTENTION_H
