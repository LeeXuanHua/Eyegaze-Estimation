import gc
import torch
import torch.nn as nn

def cosine_sim(first_v, second_v):
    dot_product = torch.dot(first_v, second_v)
    norm_f = torch.linalg.norm(first_v)
    norm_s = torch.linalg.norm(second_v)
    del first_v
    del second_v
    return torch.abs(dot_product / (norm_f * norm_s))

def prune(model, prune_ratio):
    rr = 0
    ll = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.weight.size(3)>=3 and module.weight.size(0) <= 128:
            weight_flatten = torch.flatten(module.weight, 1)
            size = module.weight.size(0)
            n_top = int(prune_ratio * size)
            similarity = torch.zeros(size)
            
            for first_i in range(size):
                for second_i in range(first_i+1, size):
                    sim = cosine_sim(weight_flatten[first_i], weight_flatten[second_i])
                    similarity[first_i] += sim / (size-1)
                    similarity[second_i] += sim / (size-1)
                    
            index = [i for i in range(size)]
            index = [x for _, x in sorted(zip(similarity, index))]
            index = index[-n_top:]
            
            with torch.no_grad():
                new_weight = module.weight
                for i in index:
                    new_weight[i] = nn.init.xavier_normal_(torch.ones(size = new_weight.size()[1:]))
                module.weight = nn.Parameter(new_weight)
            del weight_flatten
            del similarity
            gc.collect()
            ll += size
            rr += len(index)
    print(f"Pruned Filters : {rr}/{ll}")
    return model