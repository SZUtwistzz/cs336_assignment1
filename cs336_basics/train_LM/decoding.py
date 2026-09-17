from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_LM.softmax import softmax
from cs336_basics.transformer_LM.transformer_lm import transformer_LM
import torch

def decoding(prompt:str,model,tokenizer:Tokenizer,temperature,top_p,max_new_tokens,device = "cuda" if torch.cuda.is_available() else "cpu"):
    prompt_tokens = torch.tensor(tokenizer.encode(prompt),dtype = torch.long,).unsqueeze(0)
    prompt_tokens = prompt_tokens.to(device)
    endTokenId = tokenizer.encode("<|endoftext|>")[0]

    assert isinstance(max_new_tokens,int) and temperature > 0 and top_p > 0 and top_p <= 1 and max_new_tokens >= 0
    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            model_context_length = model.context_length
            
            model_prompt_tokens = prompt_tokens[:,-model_context_length:]

            if model_prompt_tokens.shape[1] == 0:
                model_prompt_tokens = torch.tensor(tokenizer.encode("<|endoftext|>"),dtype = torch.long , device=device).unsqueeze(0)
                prompt_tokens = model_prompt_tokens

            logits = model(model_prompt_tokens)

            last_logits = logits[:,-1,:]

            results = softmax(last_logits/temperature,dim=-1)

            probability,indices = torch.sort(results,dim = -1,descending=True)
            accumulate_sum = torch.cumsum(probability,dim = -1)
            final_results = accumulate_sum-probability
            ToKeep = final_results<top_p
            probability*=ToKeep
            NormForProbability = probability/probability.sum(dim = -1,keepdim =True)
            top_k_tokens = torch.multinomial(NormForProbability,num_samples=1)
            next_token = indices.gather(dim = -1,index = top_k_tokens)

            prompt_tokens = torch.cat((prompt_tokens,next_token),dim = 1)
            if next_token.item() ==endTokenId:  
                break
                
    return tokenizer.decode(prompt_tokens[0].tolist())