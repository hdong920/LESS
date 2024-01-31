# Adapted from https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_lm_eval/modify_opt.py

import torch
import torch.nn as nn

def local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget, no_padding_seq_length=None, multi_query=False):

    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        raise NotImplementedError
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(tmp_attn[:,:,padding_length:heavy_budget+recent_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,:,heavy_budget+recent_budget+padding_length:] = 0
    accumulated_attention_score[:,:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    if multi_query:
        mask_bottom = mask_bottom[:,0].unsqueeze(1) #B1SS
        accumulated_attention_score = accumulated_attention_score.sum(dim=1, keepdim=True) #B1S
    mask_bottom[:,:, padding_length:heavy_budget+recent_budget+padding_length, padding_length:heavy_budget+recent_budget+padding_length] = True

    for token_index in range(heavy_budget+recent_budget+padding_length, seq_length):
        
        tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        if multi_query:
            tmp_attn_index = tmp_attn_index.sum(dim=1, keepdim=True) #B1S
        _, tmp_topk_index = accumulated_attention_score[..., :token_index-recent_budget].topk(k=heavy_budget, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        
        mask_bottom_index[:, : , token_index-recent_budget:token_index+1] = True

        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index
    
    return mask_bottom


def get_h2o_mask(attn_weights, heavy_budget, recent_budget, multi_query):
    if heavy_budget > 0:
        mask_bottom = local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget, multi_query=multi_query) # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    if multi_query:
        ones = torch.ones_like(mask_bottom, dtype=torch.bool)
    else:
        ones = torch.ones_like(attn_weights, dtype=torch.bool)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)

    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    return mask_bottom

def get_A_mask(attn_weights, heavy_budget, recent_budget):
    A_mask = torch.ones_like(attn_weights, dtype=torch.bool)
    A_mask = torch.triu(A_mask, diagonal=-recent_budget)
    A_mask[..., :heavy_budget] = 1
    A_mask = torch.tril(A_mask, diagonal=0)
    return A_mask