
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include "model.h"
#include "kernels.h"


Transformer::Transformer(const char* model_path){
    // // 读入文件
    // FILE *file = fopen(model_path, "rb");
    
    // fseek(file, 0, SEEK_END); 
    // this -> file_size = ftell(file); 
    // fclose(file);
    
    // this -> fd = open(model_path, O_RDONLY); // open in read only mode
    
    // this -> data = (float*)(mmap(NULL, this -> file_size, PROT_READ, MAP_PRIVATE, this -> fd, 0));

    // // 解析文件
    // // 读config
    // this -> config.dim = *(int*)(this -> data);
    // this -> config.hidden_dim = *(int*)(this -> data + 1);
    // this -> config.layers = *(int*)(this -> data + 2);
    // this -> config.n_head_q = *(int*)(this -> data + 3);
    // this -> config.n_head_kv = *(int*)(this -> data + 4);
    // this -> config.vocabulary_size = *(int*)(this -> data + 5);
    // this -> config.seq_len = *(int*)(this -> data + 6);

    // // 读weights
    // int offset = 64;//头部固定256
    // this -> weights.token_embedding_table = this -> data + offset;
    // offset += this -> config.vocabulary_size * this -> config.dim;
    
    // this -> weights.rms_att_weight = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim;
    
    // this -> weights.wq = this -> data + offset;
    // int head_size = this -> config.dim / this -> config.n_head_q;
    // offset += this -> config.layers * this -> config.dim * (this -> config.n_head_q * head_size);
    
    // this -> weights.wk = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim * (this -> config.n_head_kv * head_size);
    
    // this -> weights.wv = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim * (this -> config.n_head_kv * head_size);
    
    // this -> weights.wo = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim * this -> config.dim;
    
    // this -> weights.rms_ffn_weight = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim;
    
    // this -> weights.w1 = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim * this -> config.hidden_dim;
    
    // this -> weights.w2 = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim * this -> config.hidden_dim;
    
    // this -> weights.w3 = this -> data + offset;
    // offset += this -> config.layers * this -> config.dim * this -> config.hidden_dim;
    
    // this -> weights.rms_final_weight = this -> data + offset;
    // offset += this->config.dim;

    // // 补充：分类器权重 (wcls) 逻辑
    // // 根据 Karpathy 的规定，如果 vocab_size > 0，则 wcls 和 token_embedding_table 共享内存
    // // 如果为负数，则说明它是独立的权重，需要继续往下读
    // int vocab_size = abs(this->config.vocabulary_size);
    // if (this->config.vocabulary_size > 0) {
    //     this->weights.wcls = this->weights.token_embedding_table;
    // } else {
    //     this->weights.wcls = this->data + offset;
    //     offset += vocab_size * this->config.dim;
    // }

    // // 分配运行时内存
    // int kv_dim = head_size * this->config.n_head_kv;
    
    // this->state.x.resize(this->config.dim, 0);
    // this->state.xb.resize(this->config.dim, 0);
    // this->state.xb2.resize(this->config.dim, 0);
    // this->state.hb.resize(this->config.hidden_dim, 0);
    // this->state.hb2.resize(this->config.hidden_dim, 0);
    // this->state.q.resize(this->config.dim, 0);
    // this->state.k.resize(kv_dim, 0);
    // this->state.v.resize(kv_dim, 0);
    
    // this->state.attention.resize(this->config.n_head_q * this->config.seq_len, 0);
    // this->state.logits.resize(vocab_size, 0);

    // this->state.key_cache.resize(this->config.layers * this->config.seq_len * kv_dim, 0);
    // this->state.value_cache.resize(this->config.layers * this->config.seq_len * kv_dim, 0);

    // this->config.vocabulary_size = vocab_size;

    // std::cout << this->config.layers << std::endl;

    // 1. 读取头文件获取 Config
    FILE *file = fopen(model_path, "rb");
    int header[256 / sizeof(int)] = {0};
    fread(header, sizeof(int), 256 / sizeof(int), file);
    fclose(file);

    this->config.dim = header[0];
    this->config.hidden_dim = header[1];
    this->config.layers = header[2];
    this->config.n_head_q = header[3];
    this->config.n_head_kv = header[4];
    this->config.vocabulary_size = abs(header[5]);
    this->config.seq_len = header[6];

    // 2. CPU mmap 映射文件读取权重
    int fd = open(model_path, O_RDONLY);
    size_t file_size = lseek(fd, 0, SEEK_END);
    void* mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    float* cpu_weights_start = static_cast<float*>(mapped_data) + (256 / sizeof(float));

    // 3. GPU 显存常驻分配
    size_t weights_bytes = file_size - 256;
    cudaMalloc((void**)&this->device_weights_memory, weights_bytes);
    
    cudaMemcpy(this->device_weights_memory, cpu_weights_start, weights_bytes, cudaMemcpyHostToDevice);
    
    // 释放 CPU 内存
    munmap(mapped_data, file_size);
    close(fd);

    // 4. 计算各个矩阵在 GPU 显存中的偏移量
    float* d_ptr = this->device_weights_memory;
    this->weights.token_embedding_table = d_ptr; d_ptr += config.vocabulary_size * config.dim;
    this->weights.rms_att_weight = d_ptr;        d_ptr += config.layers * config.dim;
    int head_size = config.dim / config.n_head_q;
    this->weights.wq = d_ptr; d_ptr += config.layers * config.dim * (config.n_head_q * head_size);
    this->weights.wk = d_ptr; d_ptr += config.layers * config.dim * (config.n_head_kv * head_size);
    this->weights.wv = d_ptr; d_ptr += config.layers * config.dim * (config.n_head_kv * head_size);
    this->weights.wo = d_ptr; d_ptr += config.layers * config.dim * config.dim;
    this->weights.rms_ffn_weight = d_ptr; d_ptr += config.layers * config.dim;
    this->weights.w1 = d_ptr; d_ptr += config.layers * config.dim * config.hidden_dim;
    this->weights.w2 = d_ptr; d_ptr += config.layers * config.dim * config.hidden_dim;
    this->weights.w3 = d_ptr; d_ptr += config.layers * config.dim * config.hidden_dim;
    this->weights.rms_final_weight = d_ptr; d_ptr += config.dim;
    
    if (header[5] > 0) this->weights.wcls = this->weights.token_embedding_table;
    else this->weights.wcls = d_ptr;

    // 5. 激活动态显存常驻
    int kv_dim = (config.dim * config.n_head_kv) / config.n_head_q;
    size_t run_mem_size = 0;
    run_mem_size += config.dim * 3; // x, xb, xb2
    run_mem_size += config.hidden_dim * 2; // hb, hb2
    run_mem_size += config.dim + kv_dim * 2; // q, k, v
    run_mem_size += config.n_head_q * config.seq_len; // attention
    run_mem_size += config.vocabulary_size; // logits
    run_mem_size += config.layers * config.seq_len * kv_dim * 2; // key, value cache

    cudaMalloc((void**)&this->device_run_memory, run_mem_size * sizeof(float));
    
    // 依次分配指针
    float* r_ptr = this->device_run_memory;
    state.x = r_ptr; r_ptr += config.dim;
    state.xb = r_ptr; r_ptr += config.dim;
    state.xb2 = r_ptr; r_ptr += config.dim;
    state.hb = r_ptr; r_ptr += config.hidden_dim;
    state.hb2 = r_ptr; r_ptr += config.hidden_dim;
    state.q = r_ptr; r_ptr += config.dim;
    state.k = r_ptr; r_ptr += kv_dim;
    state.v = r_ptr; r_ptr += kv_dim;
    state.attention = r_ptr; r_ptr += config.n_head_q * config.seq_len;
    state.logits = r_ptr; r_ptr += config.vocabulary_size;
    state.key_cache = r_ptr; r_ptr += config.layers * config.seq_len * kv_dim;
    state.value_cache = r_ptr;
}

Transformer::~Transformer() {
    cudaFree(device_weights_memory);
    cudaFree(device_run_memory);
}

void rms_norm(float* out, float* x, float* weight, int l){
    float rms = 0.0f;
    for(int i=0; i<l; i++){
        rms += x[i] * x[i];
    }
    rms /= l;
    rms += 1e-5f;
    rms = 1.0f / std::sqrt(rms);
    for(int i=0; i<l; i++){
        out[i] = weight[i] * x[i] * rms;
    }
};

void matmul(float* out, float* x, float* weight, int m, int n){
    // weight 是m*n矩阵， x是n维向量, weight * x
    for(int i=0;i < m;i++){
        out[i] = 0.0f;
        for(int j=0;j<n;j++){
            out[i] += weight[ i*n + j] * x[j];
        }
    }
}

float dot(float* a, float* b, int l){
    // ab点积
    float res = 0;
    for(int i=0;i<l;i++){
        res += a[i]*b[i];
    }
    return res;
}

void softmax(float* x, int size) {
    
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


std::vector<float> Transformer::forward(int token, int pos){
    // Config& config = this -> config;
    // Weights& w = this -> weights;
    // Run& s = this -> state;

    // int dim = config.dim;
    // //s.x.assign(w.token_embedding_table + token * dim, w.token_embedding_table + (token + 1) * dim);
    // memcpy(s.x.data(), w.token_embedding_table + token * dim, dim * sizeof(float));

    // int kv_dim = s.k.size();

    // int head_size = dim / config.n_head_q;
    // int q_pergroup = config.n_head_q / config.n_head_kv; //每组q的个数

    // for(int l = 0;l < config.layers;l++){

    //     // 1.rms归一化
    //     rms_norm(s.xb.data(), s.x.data(), w.rms_att_weight + l*dim, dim);
    //     // 2.计算qkv
        
    //     // matmul(s.q.data(), s.xb.data(), w.wq + l * dim * dim, dim, dim);
    //     // matmul(s.k.data(), s.xb.data(), w.wk + l * kv_dim * dim, kv_dim, dim);
    //     // matmul(s.v.data(), s.xb.data(), w.wv + l * kv_dim * dim, kv_dim, dim);

    //     matmul_cuda(s.q.data(), s.xb.data(), w.wq + l * dim * dim, dim, dim);
    //     matmul_cuda(s.k.data(), s.xb.data(), w.wk + l * kv_dim * dim, kv_dim, dim);
    //     matmul_cuda(s.v.data(), s.xb.data(), w.wv + l * kv_dim * dim, kv_dim, dim);

    //     // 3.qk RoPE编码(k存入 cache)
    //     for (int i = 0; i < dim; i+=2) {
    //         int head_dim = i % head_size;
    //         float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    //         float val = pos * freq;
    //         float fcr = cosf(val);
    //         float fci = sinf(val);
            
    //         float v0 = s.q[i];
    //         float v1 = s.q[i+1];
    //         s.q[i]   = v0 * fcr - v1 * fci;
    //         s.q[i+1] = v0 * fci + v1 * fcr;

    //         if(i < kv_dim){
    //             float v0 = s.k[i];
    //             float v1 = s.k[i+1];
    //             s.k[i]   = v0 * fcr - v1 * fci;
    //             s.k[i+1] = v0 * fci + v1 * fcr;
    //         }
            
    //     }

    //     int kv_cache_offset = l * config.seq_len * kv_dim;
    //     memcpy(s.key_cache.data() + kv_cache_offset + pos * kv_dim, s.k.data(), kv_dim* sizeof(float));
    //     memcpy(s.value_cache.data() + kv_cache_offset + pos * kv_dim, s.v.data(), kv_dim * sizeof(float));

    //     // 4.GQA
    //     for(int i=0;i<config.n_head_q;i++){
    //         int group = i/q_pergroup; //当前组数
    //         for(int j=0;j<=pos;j++){
    //             int k_pos = kv_cache_offset + j * kv_dim + group * head_size;
    //             s.attention[i*config.seq_len + j] = dot(s.q.data() + i*head_size, s.key_cache.data() + k_pos, head_size);
    //             s.attention[i*config.seq_len + j] /= sqrtf(head_size);
    //         }
    //         softmax(s.attention.data() + i*config.seq_len, pos + 1);
            
    //         for(int k = 0; k < head_size; k++) {
    //             s.xb[i*head_size + k] = 0.0f; // 初始化当前 head 的输出
    //         }

    //         for(int j=0;j<=pos;j++){
    //             int v_pos = kv_cache_offset + j * kv_dim + group * head_size;
    //             for(int k=0;k<head_size;k++){
    //                 s.xb[i*head_size + k] += s.attention[i*config.seq_len + j] * s.value_cache[v_pos + k];
    //             }
    //         }

    //     }
        
    //     matmul(s.xb2.data(), s.xb.data(), w.wo + l*dim*dim, dim, dim);

    //     // 5.残差
    //     for (int i = 0; i < dim; i++) {
    //         s.x[i] += s.xb2[i];
    //     }
    //     // 6.ffn
    //     
    //     rms_norm(s.xb.data(), s.x.data(), w.rms_ffn_weight + l*dim, dim);
        
    //     matmul(s.hb.data(), s.xb.data(), w.w1 + l*dim*config.hidden_dim, config.hidden_dim, dim);
    //     matmul(s.hb2.data(), s.xb.data(), w.w3 + l*dim*config.hidden_dim, config.hidden_dim, dim);

    //     // SwiGLU non-linearity
    //     for (int i = 0; i < config.hidden_dim; i++) {
    //         float val = s.hb[i];
    //         // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    //         val *= (1.0f / (1.0f + expf(-val)));
    //         // elementwise multiply with w3(x)
    //         val *= s.hb2[i];
    //         s.hb[i] = val;
    //     }

    //     matmul(s.xb.data(), s.hb.data(), w.w2 + l*dim*config.hidden_dim, dim, config.hidden_dim);

    //     for (int i = 0; i < dim; i++) {
    //         s.x[i] += s.xb[i];
    //     }
    // }

    // // 7.最后归一化
    // rms_norm(s.x.data(), s.x.data(), w.rms_final_weight, dim);

    // // 返回logits
    // matmul(s.logits.data(), s.x.data(), w.wcls, config.vocabulary_size, dim);
    // return s.logits;

    Config& c = this->config;
    Weights& w = this->weights;
    Run& s = this->state;

    int dim = c.dim;
    int kv_dim = (dim * c.n_head_kv) / c.n_head_q;
    int head_size = dim / c.n_head_q;

    
    launch_copy_embedding(s.x, w.token_embedding_table, token, dim);

    for (int l = 0; l < c.layers; l++) {
        // Attention 层
        launch_rmsnorm(s.xb, s.x, w.rms_att_weight + l * dim, dim);
        
        launch_matmul(s.q, s.xb, w.wq + l * dim * dim, dim, dim);
        launch_matmul(s.k, s.xb, w.wk + l * kv_dim * dim, kv_dim, dim); // 注意这里维度 m,n 传入方式
        launch_matmul(s.v, s.xb, w.wv + l * kv_dim * dim, kv_dim, dim);

        launch_rope(s.q, s.k, pos, dim, kv_dim, head_size);
        
        launch_cache_update(s.key_cache, s.value_cache, s.k, s.v, l, pos, c.seq_len, kv_dim);
        
        launch_attention(s.xb, s.q, s.key_cache, s.value_cache, s.attention, l, pos, c.seq_len, c.n_head_q, c.n_head_kv, head_size, kv_dim);
        
        launch_matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim);
        launch_add(s.x, s.xb2, dim);

        // FFN 层
        launch_rmsnorm(s.xb, s.x, w.rms_ffn_weight + l * dim, dim);
        
        launch_matmul(s.hb, s.xb, w.w1 + l * dim * c.hidden_dim, c.hidden_dim, dim);
        launch_matmul(s.hb2, s.xb, w.w3 + l * dim * c.hidden_dim, c.hidden_dim, dim);
        
        launch_swiglu(s.hb, s.hb2, c.hidden_dim);
        
        launch_matmul(s.xb, s.hb, w.w2 + l * dim * c.hidden_dim, dim, c.hidden_dim);
        launch_add(s.x, s.xb, dim);
    }

    launch_rmsnorm(s.x, s.x, w.rms_final_weight, dim);
    launch_matmul(s.logits, s.x, w.wcls, c.vocabulary_size, dim);

    // logits拷回CPU
    std::vector<float> cpu_logits(c.vocabulary_size);
    cudaMemcpy(cpu_logits.data(), s.logits, c.vocabulary_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    return cpu_logits;
}