#ifndef MODEL_H
#define MODEL_H


#include <vector>
#include <string>

struct Config{
    int dim; //每个词向量维度
    int hidden_dim; //隐藏层维度
    int layers; //层数
    int n_head_q; //每层q的头数
    int n_head_kv; //每层kv头数
    int vocabulary_size; //支持的词个数
    int seq_len; //最大序列长度
};

struct Weights{
    // 词向量表
    float* token_embedding_table;    // (vocab_size, dim)
    // rms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // qkv
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    
    float* rms_final_weight; // (dim,)
    
    float* wcls;
};

struct  Run{
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* attention;
    float* logits;
    float* key_cache;
    float* value_cache;
};
void softmax(float* x, int size);
class Transformer{
    private:
        Config config;
        Weights weights;
        Run state;
        int fd;
        float* data;
        size_t file_size;
        // 指向显存的指针
        float* device_weights_memory;
        float* device_run_memory;
    public:
        Transformer(const char* model_path);
        std::vector<float> forward(int token, int pos);
        ~Transformer();
};

#endif
