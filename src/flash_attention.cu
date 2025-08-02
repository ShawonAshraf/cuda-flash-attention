#include "flash_attention.h"
#include <iostream>
#include <cassert>

void flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    const FlashAttentionConfig& config
) {
    // Validate input dimensions
    assert(batch_size > 0);
    assert(num_heads > 0);
    assert(seq_len_q > 0);
    assert(seq_len_k > 0);
    assert(config.head_dim > 0);
    assert(config.tile_size > 0);
    
    // Calculate grid dimensions
    int grid_x = (seq_len_k + config.tile_size - 1) / config.tile_size;  // K tiles
    int grid_y = (seq_len_q + config.tile_size - 1) / config.tile_size;  // Q tiles
    int grid_z = batch_size * num_heads;  // Batch * Head
    
    // Calculate shared memory requirements
    size_t shared_mem_size = 
        4 * config.tile_size * config.head_dim * sizeof(float) +  // Q, K, V, O tiles
        config.tile_size * sizeof(TileState);                      // Softmax stats
    
    // Limit shared memory usage
    size_t max_shared_mem = 48 * 1024;  // 48KB typical limit
    if (shared_mem_size > max_shared_mem) {
        std::cerr << "Warning: Shared memory requirement (" << shared_mem_size 
                  << ") exceeds limit (" << max_shared_mem << ")" << std::endl;
        // Adjust tile size or use alternative approach
    }
    
    // Configure kernel launch parameters
    dim3 block_size(config.block_size, 4, 1);  // Optimize for memory coalescing
    dim3 grid_size(grid_x, grid_y, grid_z);
    
    // Launch kernel
    flash_attention_inner_kernel<<<grid_size, block_size, shared_mem_size>>>(
        Q, K, V, O,
        batch_size, num_heads, seq_len_q, seq_len_k,
        config.head_dim, config.softmax_scale, config.tile_size
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Alternative implementation with better memory access patterns
void flash_attention_forward_optimized(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    const FlashAttentionConfig& config
) {
    // Initialize output to zero
    size_t output_size = batch_size * num_heads * seq_len_q * config.head_dim * sizeof(float);
    CUDA_CHECK(cudaMemset(O, 0, output_size));
    
    // Launch configuration
    const int threads_per_block = 256;
    const int tile_size = 64;
    
    dim3 block_dim(16, 16);  // 256 threads
    dim3 grid_dim(
        (seq_len_k + tile_size - 1) / tile_size,
        (seq_len_q + tile_size - 1) / tile_size,
        batch_size * num_heads
    );
    
    // Shared memory calculation
    size_t shared_size = 
        4 * tile_size * config.head_dim * sizeof(float) +  // Q, K, V, partial O
        tile_size * tile_size * sizeof(float) +            // Attention matrix
        tile_size * sizeof(TileState);                     // Softmax stats
    
    // Launch kernel
    flash_attention_inner_kernel<<<grid_dim, block_dim, shared_size>>>(
        Q, K, V, O,
        batch_size, num_heads, seq_len_q, seq_len_k,
        config.head_dim, config.softmax_scale, tile_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
