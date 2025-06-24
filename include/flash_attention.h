#pragma once

#include <cuda_runtime.h>

void flash_attention_forward(
    float* q, 
    float* k, 
    float* v, 
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);
