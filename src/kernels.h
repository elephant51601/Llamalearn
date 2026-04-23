#pragma once


void launch_copy_embedding(float* d_x, float* d_table, int token, int dim);

void launch_rmsnorm(float* d_out, float* d_x, float* d_weight, int size);

void launch_matmul(float* d_out, float* d_x, float* d_weight, int m, int n);

void launch_rope(float* d_q, float* d_k, int pos, int dim, int kv_dim, int head_size);

void launch_add(float* d_a, float* d_b, int size);

void launch_swiglu(float* d_hb, float* d_hb2, int hidden_dim);

void launch_attention(float* d_xb, float* d_q, float* d_key_cache, float* d_value_cache, float* d_attention, 
                      int layer, int pos, int seq_len, int n_head_q, int n_head_kv, int head_size, int kv_dim);

void launch_cache_update(float* d_key_cache, float* d_value_cache, float* d_k, float* d_v, 
                         int layer, int pos, int seq_len, int kv_dim);