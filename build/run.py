import torch
from transformers import AutoTokenizer
import llm_engine


model_path = "../weights/stories15M.bin"
model = llm_engine.Transformer(model_path)


print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


prompt = "Once upon a time, there was a little dog"

input_ids = tokenizer.encode(prompt)

print("Prompt:", prompt, end="")

max_new_tokens = 100
temperature = 1
top_p = 0.9

for pos in range(len(input_ids) + max_new_tokens):
   
    token = input_ids[-1]
    
    # 用 Python 调用C++ 底层运算引擎
    logits_list = model.forward(token, pos) 
    
    if pos < len(input_ids) - 1:
        continue
        
    
    logits = torch.tensor(logits_list)
    logits = logits / temperature
    
    
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    
    
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    input_ids.append(next_token)
    
    
    word = tokenizer.decode([next_token])
    print(word, end=" ", flush=True)
    
    if next_token == tokenizer.eos_token_id:
        break

print("\n\nDone.")