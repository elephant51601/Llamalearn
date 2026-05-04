#include <cuda_runtime.h>
#include <iostream>
#include "kernels.h"

// 1. Embedding 拷贝核函数
__global__ void copy_embedding_kernel(float* x, float* table, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim){
        x[i] = table[token * dim + i];
    }
    
}
void launch_copy_embedding(float* d_x, float* d_table, int token, int dim) {
    int blocks = (dim + 255) / 256;
    copy_embedding_kernel<<<blocks, 256>>>(d_x, d_table, token, dim);
}

// 2. 残差连接
__global__ void add_kernel(float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) a[i] += b[i];
}
void launch_add(float* d_a, float* d_b, int size) {
    int blocks = (size + 255) / 256;
    add_kernel<<<blocks, 256>>>(d_a, d_b, size);
}

// 3. 矩阵乘法
__global__ void matmul_kernel(float* out, float* x, float* weight, int m, int n) {
    int row = blockIdx.x;
    int c = threadIdx.x;
    
    float val = 0.0f;
    for (int j = c; j < n; j+=blockDim.x) {
        val += weight[row * n + j] * x[j];
    }
    
    // 此时val是每个线程自己的和，要全部相加
    // 把每个线程算的部分和写到共享内存
    __shared__ float sdata[256];
    sdata[c] = val;
    __syncthreads();

    // 规约求和
    // for (int s = 1; s < blockDim.x; s *= 2) {
    //     if (c % (2 * s) == 0) {
    //         sdata[c] += sdata[c + s];
    //     }
    //     __syncthreads();
    // }

    for(int s = blockDim.x / 2; s>0; s >>= 1){
        if(c < s){
            sdata[c] += sdata[c + s];
        }
        __syncthreads();
    }

    if (c == 0) {
        out[row] = sdata[0];
    }
    
}
void launch_matmul(float* d_out, float* d_x, float* d_weight, int m, int n) {
    matmul_kernel<<<m, 256>>>(d_out, d_x, d_weight, m, n);
}

// 4. RMSNorm
__global__ void rmsnorm_kernel(float* out, float* x, float* weight, int size) {
    
    int tid = threadIdx.x;

    __shared__ float s_sum[256];
    __shared__ float sum;
    
    float local_sum = 0.0f;
    for(int t=tid; t<size; t+= blockDim.x){
        local_sum += x[t] * x[t];
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    
    for(int s=blockDim.x / 2;s > 0;s >>= 1){
        if(tid < s){
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
        sum = 1.0f / sqrtf(s_sum[0] / size + 1e-5f);
    __syncthreads();
    
    for(int t=tid; t<size; t+= blockDim.x){
        out[t] = weight[t] * x[t] * sum;
    }
}
void launch_rmsnorm(float* d_out, float* d_x, float* d_weight, int size) {
    // int blocks = (size + 255) / 256;
    rmsnorm_kernel<<<1, 256>>>(d_out, d_x, d_weight, size);
}

// 5. SwiGLU 激活
__global__ void swiglu_kernel(float* hb, float* hb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}
void launch_swiglu(float* d_hb, float* d_hb2, int hidden_dim) {
    int blocks = (hidden_dim + 255) / 256;
    swiglu_kernel<<<blocks, 256>>>(d_hb, d_hb2, hidden_dim);
}

// 6. RoPE 旋转位置编码
__global__ void rope_kernel(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // 处理 i 和 i+1
    if (i < dim) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        
        float v0 = q[i];
        float v1 = q[i+1];
        q[i]   = v0 * fcr - v1 * fci;
        q[i+1] = v0 * fci + v1 * fcr;

        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i+1];
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }
}
void launch_rope(float* d_q, float* d_k, int pos, int dim, int kv_dim, int head_size) {
    int blocks = (dim / 2 + 255) / 256;
    rope_kernel<<<blocks, 256>>>(d_q, d_k, pos, dim, kv_dim, head_size);
}

// 7. 更新 KV Cache 
__global__ void cache_update_kernel(float* key_cache, float* value_cache, float* k, float* v, 
                                    int kv_cache_offset, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        key_cache[kv_cache_offset + i] = k[i];
        value_cache[kv_cache_offset + i] = v[i];
    }
}
void launch_cache_update(float* d_key_cache, float* d_value_cache, float* d_k, float* d_v, 
                         int layer, int pos, int seq_len, int kv_dim) {
    int offset = layer * seq_len * kv_dim + pos * kv_dim;
    int blocks = (kv_dim + 255) / 256;
    cache_update_kernel<<<blocks, 256>>>(d_key_cache, d_value_cache, d_k, d_v, offset, kv_dim);
}


// 8. GQA 注意力机制
__global__ void attention_kernel(float* xb, float* q, float* key_cache, float* value_cache, float* att, 
                                 int kv_cache_offset, int pos, int seq_len, int n_head_q, int q_pergroup, int head_size, int kv_dim) {
    // 让每个 Block 负责一个 Q
    int h = blockIdx.x; 
    int tid = threadIdx.x;
    if (h >= n_head_q) return;
    
    int group = h / q_pergroup;
    float* my_q = q + h * head_size;
    float* my_att = att + h * seq_len;
    float* my_xb = xb + h * head_size;
    
   
    for(int t=tid; t<pos; t+= blockDim.x){
        float* my_k = key_cache + kv_cache_offset + t * kv_dim + group * head_size;

        // 直接算完整点积
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += my_q[i] * my_k[i];
        }

        my_att[t] = score / sqrtf((float)head_size);
        
    }
    __syncthreads();

    // int warp_id = tid / 32;    // 哪个warp
    // int lane_id = tid % 32;    // warp内编号 0~31

    // int num_warp = blockDim.x / 32;  // 一个block 8个warp
    // int t_start = warp_id;

    // // 一个 warp 处理一个 t
    // for (int t = t_start; t <= pos; t += num_warp)
    // {
    //     float* k = key_cache + kv_cache_offset + t * kv_dim + group * head_size;

    //     // warp 内并行点积
    //     float sum = 0.0f;
    //     for (int i = lane_id; i < head_size; i += 32) {
    //         sum += my_q[i] * k[i];
    //     }

    //     // warp 内规约
    //     for (int s = 1; s < 32; s *= 2) {
    //         sum += __shfl_down_sync(0xffffffff, sum, s);
    //     }

    //     // lane 0 写结果
    //     if (lane_id == 0) {
    //         my_att[t] = sum / sqrtf((float)head_size);
    //     }
    // }
    // __syncthreads();


    // Softmax
    
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    float local_max = -1e9;  
    for (int t = tid; t <= pos; t += blockDim.x) {
        local_max = fmaxf(local_max, my_att[t]);
    }
    s_max[tid] = local_max;
    __syncthreads();
    
    // 规约求最大值
    
    for(int s = blockDim.x / 2; s > 0; s>>= 1){
        if(tid<s){
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    
    float local_sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        my_att[t] = expf(my_att[t] - s_max[0]);
        local_sum += my_att[t];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s>>= 1){
        if(tid < s){
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    for(int t = tid; t<=pos; t+=blockDim.x){
        my_att[t] /= s_sum[0];
    }
    __syncthreads();

    // 加权求和v
    for (int i = tid; i < head_size; i+=blockDim.x){
        my_xb[i] = 0.0f;
    } 
    __syncthreads();
   

    for(int i = tid;i < head_size;i++){
        float val = 0.0f;
        for(int t = 0; t<= pos; t++){
            float* my_v = value_cache + kv_cache_offset + t*kv_dim + group * head_size;
            float a = my_att[t];
            val += a * my_v[i];
        }
        my_xb[i] = val;
    }
}
void launch_attention(float* d_xb, float* d_q, float* d_key_cache, float* d_value_cache, float* d_attention, 
                      int layer, int pos, int seq_len, int n_head_q, int n_head_kv, int head_size, int kv_dim) {
    int kv_cache_offset = layer * seq_len * kv_dim;
    int q_pergroup = n_head_q / n_head_kv;
    // 分配 n_head_q 个线程块，每个块处理一个头
    attention_kernel<<<n_head_q, 256>>>(d_xb, d_q, d_key_cache, d_value_cache, d_attention, 
                                      kv_cache_offset, pos, seq_len, n_head_q, q_pergroup, head_size, kv_dim);
}
