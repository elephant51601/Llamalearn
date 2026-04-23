#include <iostream>
#include "model.h"



int sample(std::vector<float>& probabilities) {
    float threshold = 0.01f; 
    float sum = 0.0f;
    for (int i = 0; i < probabilities.size(); i++) {
        if (probabilities[i] < threshold) {
            probabilities[i] = 0.0f;
        }
        sum += probabilities[i];
    }
    
    
    for (int i = 0; i < probabilities.size(); i++) {
        probabilities[i] /= sum;
    }

    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < probabilities.size(); i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return probabilities.size() - 1;
}

std::vector<std::string> load_tokenizer(std::string path, int vocab_size) {
    std::vector<std::string> vocab(vocab_size);
    FILE *file = fopen(path.c_str(), "rb");
    if (!file) {
        std::cerr << "未找到 tokenizer.bin" << std::endl;
        return vocab;
    }
    int max_token_length;
    fread(&max_token_length, sizeof(int), 1, file);
    for (int i = 0; i < vocab_size; i++) {
        float score;
        fread(&score, sizeof(float), 1, file);
        int len;
        fread(&len, sizeof(int), 1, file);
        char buffer[256]; 
        fread(buffer, 1, len, file);
        buffer[len] = '\0';
        vocab[i] = std::string(buffer);
    }
    fclose(file);
    return vocab;
}

int main(){
    srand(time(NULL));
    std::cout << "hello";
    const char* path = "../weights/stories15M.bin";
    Transformer transformer(path);

    // 开始生成
    int next_token = 3; 
    int max_seq_len = 256; //生成 256 个词

    std::cout << "Generating story...\n";
    std::vector<std::string> vocab = load_tokenizer("../src/tokenizer.bin", 32000);

    for (int pos = 0; pos < max_seq_len; pos++) {
        
        std::vector<float> logits = transformer.forward(next_token, pos);

        // 加上温度缩放
        float temperature = 0.5f;
        for (int i = 0; i < logits.size(); i++) {
            logits[i] /= temperature;
        }

       
        softmax(logits.data(), logits.size());
        next_token = sample(logits);

        
        std::string text = vocab[next_token];

        // 处理 Llama 特有的下划线空格 (字节码为 e2 96 81)
        if (text.length() >= 3 && text[0] == (char)0xe2 && text[1] == (char)0x96 && text[2] == (char)0x81) {
            text = " " + text.substr(3); // 替换成我们键盘上打出来的正常空格
        }

        // 处理特殊的换行符
        if (text == "<0x0A>") {
            text = "\n";
        }

        
        std::cout << text;
        fflush(stdout);
        
        
        if (next_token == 2) break; 
    }

    std::cout << "\nGeneration done." << std::endl;
    return 0;

}