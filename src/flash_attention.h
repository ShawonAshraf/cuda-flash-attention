#ifndef FLASH_ATTENTION_IMPL_H
#define FLASH_ATTENTION_IMPL_H

#include "../include/flash_attention.h"
#include <cuda_runtime.h>

// Shared memory structure for online softmax
struct TileState {
    float max_val;
    float sum_val;
};

// Online softmax update function
__device__ __forceinline__ void online_softmax_update(
    float& max_val,
    float& sum_val,
    float new_val
) {
    float old_max = max_val;
    max_val = fmaxf(max_val, new_val);
    
    if (max_val <= -1e10f) {
        sum_val = 0.0f;
    } else {
        sum_val = sum_val * expf(old_max - max_val) + expf(new_val - max_val);
    }
}

// Online softmax finalize function
__device__ __forceinline__ float online_softmax_finalize(
    float val,
    float max_val,
    float sum_val
) {
    if (sum_val == 0.0f) return 0.0f;
    return expf(val - max_val) / sum_val;
}

// Flash Attention forward kernel with tiling
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim,
    const float softmax_scale,
    const int tile_size
) {
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    
    // Calculate batch and head indices
    int batch_idx = blockIdx.z / num_heads;
    int head_idx = blockIdx.z % num_heads;
    
    if (batch_idx >= batch_size) return;
    
    // Calculate global indices
    int q_tile_start = blockIdx.y * tile_size;
    int k_tile_start = blockIdx.x * tile_size;
    
    // Shared memory layout
    float* sQ = shared_mem;                                    // [tile_size, head_dim]
    float* sK = shared_mem + tile_size * head_dim;            // [tile_size, head_dim]
    float* sV = shared_mem + 2 * tile_size * head_dim;        // [tile_size, head_dim]
    float* sO = shared_mem + 3 * tile_size * head_dim;        // [tile_size, head_dim]
    TileState* stats = (TileState*)(shared_mem + 4 * tile_size * head_dim); // [tile_size]
    
    // Thread indices
    int tid = threadIdx.x;
    int q_idx = threadIdx.y;
    int k_idx = threadIdx.x;
    
    // Calculate base offsets
    int q_base = batch_idx * num_heads * seq_len_q * head_dim + 
                 head_idx * seq_len_q * head_dim;
    int k_base = batch_idx * num_heads * seq_len_k * head_dim + 
                 head_idx * seq_len_k * head_dim;
    int o_base = q_base;
    
    // Load Q tile to shared memory
    for (int i = q_idx; i < tile_size && q_tile_start + q_idx < seq_len_q; i += blockDim.y) {
        for (int j = tid; j < head_dim; j += blockDim.x) {
            int q_global_idx = q_base + (q_tile_start + i) * head_dim + j;
            sQ[i * head_dim + j] = Q[q_global_idx];
        }
    }
    
    // Load K and V tiles to shared memory
    for (int i = q_idx; i < tile_size && k_tile_start + q_idx < seq_len_k; i += blockDim.y) {
        for (int j = tid; j < head_dim; j += blockDim.x) {
            int k_global_idx = k_base + (k_tile_start + i) * head_dim + j;
            int v_global_idx = k_global_idx; // Same layout
            sK[i * head_dim + j] = K[k_global_idx];
            sV[i * head_dim + j] = V[v_global_idx];
        }
    }
    
    __syncthreads();
    
    // Initialize output tile
    for (int i = q_idx; i < tile_size; i += blockDim.y) {
        for (int j = tid; j < head_dim; j += blockDim.x) {
            sO[i * head_dim + j] = 0.0f;
        }
    }
    
    // Initialize softmax statistics
    if (q_idx < tile_size && tid == 0) {
        stats[q_idx].max_val = -1e20f;
        stats[q_idx].sum_val = 0.0f;
    }
    
    __syncthreads();
    
    // Compute attention scores and apply online softmax
    for (int i = q_idx; i < tile_size && q_tile_start + q_idx < seq_len_q; i += blockDim.y) {
        int q_seq_idx = q_tile_start + i;
        
        // Compute attention scores for this query against all keys in tile
        for (int j = tid; j < tile_size && k_tile_start + tid < seq_len_k; j += blockDim.x) {
            int k_seq_idx = k_tile_start + j;
            
            // Compute dot product Q[i] * K[j]
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += sQ[i * head_dim + d] * sK[j * head_dim + d];
            }
            score *= softmax_scale;
            
            // Update online softmax statistics
            online_softmax_update(stats[i].max_val, stats[i].sum_val, score);
        }
    }
    
    __syncthreads();
    
    // Finalize softmax and compute weighted sum with values
    for (int i = q_idx; i < tile_size && q_tile_start + q_idx < seq_len_q; i += blockDim.y) {
        // Recompute attention scores and apply softmax
        for (int j = tid; j < tile_size && k_tile_start + tid < seq_len_k; j += blockDim.x) {
            int k_seq_idx = k_tile_start + j;
            
            // Recompute dot product Q[i] * K[j]
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += sQ[i * head_dim + d] * sK[j * head_dim + d];
            }
            score *= softmax_scale;
            
            // Apply online softmax
            float softmax_val = online_softmax_finalize(score, stats[i].max_val, stats[i].sum_val);
            
            // Accumulate weighted sum: O += softmax_val * V
            for (int d = 0; d < head_dim; d++) {
                atomicAdd(&sO[i * head_dim + d], softmax_val * sV[j * head_dim + d]);
            }
        }
    }
    
    __syncthreads();
    
    // Write output tile to global memory
    for (int i = q_idx; i < tile_size && q_tile_start + q_idx < seq_len_q; i += blockDim.y) {
        for (int j = tid; j < head_dim; j += blockDim.x) {
            int o_global_idx = o_base + (q_tile_start + i) * head_dim + j;
            O[o_global_idx] = sO[i * head_dim + j];
        }
    }
}

// Optimized kernel for inner loop computation
__global__ void flash_attention_inner_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim,
    const float softmax_scale,
    const int tile_size
) {
    extern __shared__ float shared_mem[];
    
    int batch_head_idx = blockIdx.z;
    int batch_idx = batch_head_idx / num_heads;
    int head_idx = batch_head_idx % num_heads;
    
    if (batch_idx >= batch_size) return;
    
    int q_tile_start = blockIdx.y * tile_size;
    int k_tile_start = blockIdx.x * tile_size;
    
    // Shared memory layout
    float* sQ = shared_mem;
    float* sK = shared_mem + tile_size * head_dim;
    float* sV = shared_mem + 2 * tile_size * head_dim;
    float* sAttn = shared_mem + 3 * tile_size * head_dim; // Attention weights
    TileState* stats = (TileState*)(shared_mem + 3 * tile_size * head_dim + tile_size * tile_size);
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    
    int q_base = batch_idx * num_heads * seq_len_q * head_dim + head_idx * seq_len_q * head_dim;
    int k_base = batch_idx * num_heads * seq_len_k * head_dim + head_idx * seq_len_k * head_dim;
    int o_base = q_base;
    
    // Load tiles to shared memory
    for (int idx = tid; idx < tile_size * head_dim; idx += total_threads) {
        int q_local_idx = idx / head_dim;
        int q_dim_idx = idx % head_dim;
        int k_local_idx = q_local_idx;
        int k_dim_idx = q_dim_idx;
        
        if (q_tile_start + q_local_idx < seq_len_q) {
            sQ[idx] = Q[q_base + (q_tile_start + q_local_idx) * head_dim + q_dim_idx];
        } else {
            sQ[idx] = 0.0f;
        }
        
        if (k_tile_start + k_local_idx < seq_len_k) {
            sK[idx] = K[k_base + (k_tile_start + k_local_idx) * head_dim + k_dim_idx];
            sV[idx] = V[k_base + (k_tile_start + k_local_idx) * head_dim + k_dim_idx];
        } else {
            sK[idx] = 0.0f;
            sV[idx] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Initialize output tile
    for (int idx = tid; idx < tile_size * head_dim; idx += total_threads) {
        int q_local_idx = idx / head_dim;
        if (q_tile_start + q_local_idx < seq_len_q) {
            ((float*)shared_mem)[4 * tile_size * head_dim + idx] = 0.0f;
        }
    }
    
    // Initialize softmax statistics
    for (int i = tid; i < tile_size; i += total_threads) {
        stats[i].max_val = -1e20f;
        stats[i].sum_val = 0.0f;
    }
    
    __syncthreads();
    
    // Compute attention scores with tiling
    for (int i = threadIdx.y; i < tile_size && q_tile_start + i < seq_len_q; i += blockDim.y) {
        float max_val = -1e20f;
        float sum_val = 0.0f;
        
        // First pass: compute max and sum for online softmax
        for (int j = threadIdx.x; j < tile_size && k_tile_start + j < seq_len_k; j += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += sQ[i * head_dim + d] * sK[j * head_dim + d];
            }
            score *= softmax_scale;
            
            float old_max = max_val;
            max_val = fmaxf(max_val, score);
            if (max_val <= -1e10f) {
                sum_val = 0.0f;
            } else {
                sum_val = sum_val * expf(old_max - max_val) + expf(score - max_val);
            }
        }
        
        // Reduce max and sum across threads
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            float temp_max = __shfl_down_sync(0xffffffff, max_val, stride);
            float temp_sum = __shfl_down_sync(0xffffffff, sum_val, stride);
            
            if (stride < blockDim.x) {
                float old_max = max_val;
                max_val = fmaxf(max_val, temp_max);
                if (max_val <= -1e10f) {
                    sum_val = 0.0f;
                } else {
                    sum_val = sum_val * expf(old_max - max_val) + 
                              temp_sum * expf(temp_max - max_val);
                }
            }
        }
        
        if (threadIdx.x == 0) {
            stats[i].max_val = max_val;
            stats[i].sum_val = sum_val;
        }
    }
    
    __syncthreads();
    
    // Compute final attention and accumulate output
    for (int i = threadIdx.y; i < tile_size && q_tile_start + i < seq_len_q; i += blockDim.y) {
        for (int j = threadIdx.x; j < tile_size && k_tile_start + j < seq_len_k; j += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += sQ[i * head_dim + d] * sK[j * head_dim + d];
            }
            score *= softmax_scale;
            
            float softmax_val = 0.0f;
            if (stats[i].sum_val > 0.0f) {
                softmax_val = expf(score - stats[i].max_val) / stats[i].sum_val;
            }
            
            // Accumulate weighted sum
            float* output_tile = (float*)(shared_mem + 4 * tile_size * head_dim);
            for (int d = 0; d < head_dim; d++) {
                atomicAdd(&output_tile[i * head_dim + d], softmax_val * sV[j * head_dim + d]);
            }
        }
    }
    
    __syncthreads();
    
    // Write output back to global memory
    float* output_tile = (float*)(shared_mem + 4 * tile_size * head_dim);
    for (int idx = tid; idx < tile_size * head_dim; idx += total_threads) {
        int q_local_idx = idx / head_dim;
        int dim_idx = idx % head_dim;
        
        if (q_tile_start + q_local_idx < seq_len_q) {
            int global_idx = o_base + (q_tile_start + q_local_idx) * head_dim + dim_idx;
            O[global_idx] = output_tile[idx];
        }
    }
}

#endif // FLASH_ATTENTION_IMPL_H
