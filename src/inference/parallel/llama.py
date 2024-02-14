# Adapted from https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_lm_eval/modify_llama.py

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb
from masks import local_heavy_hitter_mask_nonoverlap

__all__ = [
    'convert_kvcache_llama_sparse', 
    'convert_kvcache_llama_less' 
    'LlamaAttentionSparse',
    'LlamaAttentionLESS']


class LlamaAttentionSparse(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.heavy_budget = config.heavy_count
        self.recent_budget = config.recent_count
        self.fix_heavy_to_initial_tokens = config.fix_heavy_to_initial_tokens
        

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        ### Heavy + Recent
        heavy_budget = self.heavy_budget
        recent_budget = self.recent_budget

        # Heavy Hitter Mask
        if heavy_budget > 0:
            if self.fix_heavy_to_initial_tokens:
                # Lambda Masking
                mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
                mask_bottom[..., :heavy_budget] = True
            else:
                # H2O
                mask_bottom = local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)

        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        del ones, key_states, hidden_states
        torch.cuda.empty_cache()
        attn_weights[~mask_bottom] = torch.min(attention_mask)
        del mask_bottom
        torch.cuda.empty_cache()

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




class LlamaAttentionLESS(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.heavy_budget = config.heavy_count
        self.recent_budget = config.recent_count
        self.fix_heavy_to_initial_tokens = config.fix_heavy_to_initial_tokens

        self.ker_dim = config.kernel_hidden_size
        self.ker_hid = config.ker_hid
        self.kernel_f = F.gelu
        a = math.sqrt(3/(self.head_dim + self.ker_dim))
        self.kernel_q_mat1 = torch.nn.init.uniform_(torch.empty(self.num_heads, self.head_dim, self.ker_hid), a=-a, b=a)
        self.kernel_k_mat1 = torch.nn.init.uniform_(torch.empty(self.num_heads, self.head_dim, self.ker_hid), a=-a, b=a)

        a = math.sqrt(3/(self.ker_hid + self.ker_dim))
        self.kernel_q_mat2 = torch.nn.init.uniform_(torch.empty(self.num_heads, self.ker_hid, self.ker_dim), a=-a, b=a)
        self.kernel_k_mat2 = torch.nn.init.uniform_(torch.empty(self.num_heads, self.ker_hid, self.ker_dim), a=-a, b=a)

        self.kernel_q_mat1 = nn.Parameter(self.kernel_q_mat1, requires_grad=True)
        self.kernel_k_mat1 = nn.Parameter(self.kernel_k_mat1, requires_grad=True)
        self.kernel_q_mat2 = nn.Parameter(self.kernel_q_mat2, requires_grad=True)
        self.kernel_k_mat2 = nn.Parameter(self.kernel_k_mat2, requires_grad=True)
        self.ker_act = F.gelu

        self.scalingD = nn.Parameter(torch.ones(1, self.num_heads, 1, self.ker_dim) * 1e-4, requires_grad=True)

        a = math.sqrt(6/(2 * self.ker_dim))
        self.interaction_k = torch.nn.init.uniform_(torch.empty(self.num_heads, self.ker_dim, self.ker_dim), a=-a, b=a)
        self.interaction_k = nn.Parameter(self.interaction_k, requires_grad=True)
        self.scalingD2 = nn.Parameter(torch.ones(1, self.num_heads, 1, self.ker_dim) * 1e-4, requires_grad=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_mask = attention_mask.to(query_states.dtype)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        heavy_budget = self.heavy_budget
        recent_budget = self.recent_budget

        # Sparse
        if heavy_budget > 0:
            if self.fix_heavy_to_initial_tokens:
                mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
                mask_bottom[..., :heavy_budget] = True
            else:
                mask_bottom = local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)


        sparse_attn_mask = torch.tril(mask_bottom, diagonal=0)
        attn_weights = (sparse_attn_mask * attn_weights) + ((~sparse_attn_mask)*  torch.finfo(attn_weights.dtype).min)

        sparse_attn_weights = attn_weights

        sparse_norms_lse = torch.logsumexp(sparse_attn_weights.to(torch.float32), -1, keepdim=True)
        lr_attn_mask = torch.logical_and(attention_mask > torch.min(attention_mask), ~sparse_attn_mask)

        # Low Rank
        query_states_ker = self.ker_act(torch.einsum('bhsd,hde->bhse', query_states, self.kernel_q_mat1))
        key_states_ker = self.ker_act(torch.einsum('bhsd,hde->bhse', key_states, self.kernel_k_mat1))
        query_states_ker = self.kernel_f(torch.einsum('bhsd,hde->bhse', query_states_ker, self.kernel_q_mat2))
        key_states_ker = torch.abs(self.scalingD) * self.kernel_f(torch.einsum('bhsd,hde->bhse', key_states_ker, self.kernel_k_mat2))
        key_states_ker = key_states_ker + torch.einsum('bhsd,hde->bhse', key_states_ker, self.interaction_k) * self.scalingD2

        lr_attn_weights = lr_attn_mask * torch.matmul(query_states_ker.abs(), key_states_ker.abs().transpose(2, 3)).to(torch.float32) #B, H, S, S
        
        lr_norms_lse = torch.log(lr_attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Synthesis
        norms_lse = torch.logaddexp(lr_norms_lse, sparse_norms_lse)
            
        attn_weights = torch.log(lr_attn_weights  + 1e-6)
        attn_weights = (attn_weights * lr_attn_mask) + (sparse_attn_weights * (~lr_attn_mask))
        attn_weights = torch.exp(attn_weights - norms_lse)
        
        # Continue as normal
        attn_output = torch.matmul(attn_weights.to(query_states.dtype), value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



def convert_kvcache_llama_sparse(model, config):
    def change_class(model, config):
        for name, module in reversed(model._modules.items()):

            if len(list(module.children())) > 0:
                model._modules[name] = change_class(module, config)

            if isinstance(module, LlamaAttention):
                model._modules[name] = LlamaAttentionSparse(config)

        return model

    checkpoint = copy.deepcopy(model.state_dict())
    model = change_class(model, config)
    model.load_state_dict(checkpoint)
    return model

def convert_kvcache_llama_less(model, config, path_func):
    def change_class(model, config):
        for name, module in reversed(model._modules.items()):

            if len(list(module.children())) > 0:
                model._modules[name] = change_class(module, config)

            if isinstance(module, LlamaAttention):
                model._modules[name] = LlamaAttentionLESS(config)

        return model
    checkpoint = copy.deepcopy(model.state_dict())
    model = change_class(model, config)
    device = model.device
    for li, l in enumerate(model.model.layers):
        loaded_data = torch.load(path_func(li))
        layer_loaded_data = loaded_data['model_state_dict']

        # Can merge scalings into linear layers
        checkpoint[f'model.layers.{li}.self_attn.kernel_q_mat1'] = layer_loaded_data['kernel_q_mat1']
        checkpoint[f'model.layers.{li}.self_attn.kernel_k_mat1'] = layer_loaded_data['kernel_k_mat1']
        checkpoint[f'model.layers.{li}.self_attn.kernel_q_mat2'] = layer_loaded_data['kernel_q_mat2']
        checkpoint[f'model.layers.{li}.self_attn.kernel_k_mat2'] = layer_loaded_data['kernel_k_mat2']
        checkpoint[f'model.layers.{li}.self_attn.scalingD'] = layer_loaded_data['scalingD']
        checkpoint[f'model.layers.{li}.self_attn.interaction_k'] = layer_loaded_data['interaction_k']
        checkpoint[f'model.layers.{li}.self_attn.scalingD2'] = layer_loaded_data['scalingD2']
    model.load_state_dict(checkpoint)
    return model.to(device)

