#include "flash_attention.h"
#include "utils.h"
#include <iostream>
#include <cmath>

__global__ void flash_attention_kernel(
    const float* q, 
    const float* k, 
    const float* v, 
    float* output,
    int seq_len,
    int head_dim
) {
    // This is a simplified kernel and doesn't implement the full FlashAttention algorithm yet.
    // It calculates attention for one block of Q.
    // A full implementation requires iterating through blocks of K and V.

    const int block_size = 256;
    const int thread_id = threadIdx.x;
    
    // Pointers to the current block in Q, K, V
    const float* q_block = q + blockIdx.x * seq_len * head_dim;
    const float* k_block = k + blockIdx.x * seq_len * head_dim;
    const float* v_block = v + blockIdx.x * seq_len * head_dim;
    float* output_block = output + blockIdx.x * seq_len * head_dim;

    // --- Tiling constants ---
    const int B_r = 256; // Block size for rows (Q)
    const int B_c = 64;  // Block size for columns (K, V)

    // --- Shared memory for a block of Q ---
    extern __shared__ float q_s[];
    
    // Each thread loads one element of Q into shared memory
    for (int i = threadIdx.x; i < B_r * head_dim; i += blockDim.x) {
        q_s[i] = q_block[i];
    }
    __syncthreads();

    // --- Online softmax state ---
    float l_i[B_r]; // running max
    float m_i[B_r]; // running sum of exp
    float o_i[B_r * head_dim]; // output accumulator

    for(int i = 0; i < B_r; ++i) {
        m_i[i] = -1e20f;
        l_i[i] = 0.0f;
    }
    for(int i = 0; i < B_r * head_dim; ++i) {
        o_i[i] = 0.0f;
    }

    // --- Loop over blocks of K and V ---
    for (int j = 0; j < seq_len; j += B_c) {
        // --- Load a block of K and V into shared memory ---
        float k_s[B_c * head_dim];
        float v_s[B_c * head_dim];
        // This is a simplified load, a real implementation would be more optimized
        for (int i = threadIdx.x; i < B_c * head_dim; i+= blockDim.x) {
            k_s[i] = k_block[j * head_dim + i];
            v_s[i] = v_block[j * head_dim + i];
        }
        __syncthreads();

        // --- Compute S_ij = Q_i * K_j^T ---
        float s_ij[B_r * B_c];
        // This is a simplified matmul, a real implementation would be more optimized
        for(int r = threadIdx.x; r < B_r; r += blockDim.x) {
            for(int c = 0; c < B_c; ++c) {
                float dot = 0.0f;
                for(int d = 0; d < head_dim; ++d) {
                    dot += q_s[r * head_dim + d] * k_s[c * head_dim + d];
                }
                s_ij[r * B_c + c] = dot / sqrtf(head_dim);
            }
        }
        __syncthreads();

        // --- Online softmax update ---
        float m_ij[B_r], l_ij[B_r];
        for (int r = threadIdx.x; r < B_r; r += blockDim.x) {
            float m_i_old = m_i[r];
            m_ij[r] = -1e20f;
            for(int c = 0; c < B_c; ++c) {
                if(s_ij[r * B_c + c] > m_ij[r]) {
                    m_ij[r] = s_ij[r * B_c + c];
                }
            }
            float new_m_i = fmaxf(m_i[r], m_ij[r]);
            l_ij[r] = expf(m_i[r] - new_m_i) * l_i[r];
            for(int c = 0; c < B_c; ++c) {
                 l_ij[r] += expf(s_ij[r * B_c + c] - new_m_i);
            }
            m_i[r] = new_m_i;
            l_i[r] = l_ij[r];
        }
        __syncthreads();
        
        // --- Update output O ---
        // P_ij = exp(S_ij - m_i_new)
        float p_ij[B_r * B_c];
        for (int r = threadIdx.x; r < B_r; r += blockDim.x) {
            for(int c = 0; c < B_c; ++c) {
                p_ij[r * B_c + c] = expf(s_ij[r * B_c + c] - m_i[r]);
            }
        }
        __syncthreads();

        // Rescale old output accumulator O
        for (int r = threadIdx.x; r < B_r; r += blockDim.x) {
            float scale = expf(m_i_old - m_i[r]);
            for(int d = 0; d < head_dim; ++d) {
                o_i[r * head_dim + d] *= scale;
            }
        }
        __syncthreads();
        
        // O += P_ij * V_j
        for(int r = threadIdx.x; r < B_r; r += blockDim.x) {
            for(int d = 0; d < head_dim; ++d) {
                float o_update = 0.0f;
                for(int c = 0; c < B_c; ++c) {
                    o_update += p_ij[r * B_c + c] * v_s[c * head_dim + d];
                }
                o_i[r * head_dim + d] += o_update;
            }
        }
        __syncthreads();
    }
    
    // --- Final write to HBM ---
    for(int i = threadIdx.x; i < B_r; i += blockDim.x) {
        float inv_l_i = 1.0f / l_i[i];
        for(int d = 0; d < head_dim; ++d) {
            output_block[i * head_dim + d] = o_i[i * head_dim + d] * inv_l_i;
        }
    }
}

void flash_attention_forward(
    float* q, 
    float* k, 
    float* v, 
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Grid and block dimensions
    dim3 grid(batch_size * num_heads, 1, 1);
    dim3 block(256, 1, 1); // 256 threads per block
    
    // Shared memory size
    int shared_mem_size = 256 * head_dim * sizeof(float);

    flash_attention_kernel<<<grid, block, shared_mem_size>>>(q, k, v, output, seq_len, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
