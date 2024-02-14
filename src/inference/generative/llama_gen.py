# Adapted from https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_hh/modify_llama.py

import copy
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb


__all__ = ['convert_kvcache_llama_sparse', 'LlamaAttentionSparse', 'convert_kvcache_llama_less', 'LlamaAttentionLESS']

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
        self.cache_budget = self.heavy_budget + self.recent_budget
        
        self.attention_masks_next = None 
        self.previous_scores = None
        self.fix_heavy_to_initial_tokens = config.fix_heavy_to_initial_tokens
        
    def _reset_masks(self):
        self.attention_masks_next = None 
        self.previous_scores = None
            
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
        assert bsz == 1
        if past_key_value is None:
            self._reset_masks()
            pass
        
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

        
        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
        
        # Accumulate attention scores
        if not self.previous_scores == None:
            current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
        
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum #(heads, k-tokens)
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)
        attn_tokens_all = self.previous_scores.shape[-1]
    
        if attn_tokens_all > self.cache_budget:
            # activate most recent k-cache
            if not self.recent_budget == 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                # activate historical best self.cache_budget - self.recent_budget tokens.
                # self.previous_scores # (k-Cache - 1)
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                if self.fix_heavy_to_initial_tokens:
                    #Lambda
                    attn_mask[..., :self.heavy_budget] = 1
                else:
                    #H2O
                    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                    attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)
        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        
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

        # self.fixed_count = config.fixed_sparse_count
        self.heavy_budget = config.heavy_count
        self.recent_budget = config.recent_count
        self.cache_budget = self.heavy_budget + self.recent_budget
            
        self.attention_masks_next = None 
        self.previous_scores = None
        
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

        self.H = torch.zeros((1, self.num_heads, self.ker_dim, self.head_dim))
        self.z = torch.zeros((1, self.num_heads, self.ker_dim, 1))
        

        self.scalingD = nn.Parameter(torch.ones(1, self.num_heads, 1, self.ker_dim) * 1e-4, requires_grad=True)
        
        a = math.sqrt(3/(2 * self.ker_dim))
        self.interaction_k = torch.nn.init.uniform_(torch.empty(self.num_heads, self.ker_dim, self.ker_dim), a=-a, b=a)
        self.interaction_k = nn.Parameter(self.interaction_k, requires_grad=True)
        self.scalingD2 = nn.Parameter(torch.ones(1, self.num_heads, 1, self.ker_dim) * 1e-4, requires_grad=True)
        
        self.fix_heavy_to_initial_tokens = config.fix_heavy_to_initial_tokens

    def _reset_masks(self):
        self.attention_masks_next = None 
        self.previous_scores = None
        self.H = torch.zeros((1, self.num_heads, self.ker_dim, self.head_dim))
        self.z = torch.zeros((1, self.num_heads, self.ker_dim, 1))

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
        assert bsz == 1

        if past_key_value is None:
            self._reset_masks()

        self.H = self.H.to(hidden_states.device)
        self.z = self.z.to(hidden_states.device)
        
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

        
        query_states_ker = self.ker_act(torch.einsum('bhsd,hde->bhse', query_states, self.kernel_q_mat1))
        query_states_ker = self.kernel_f(torch.einsum('bhsd,hde->bhse', query_states_ker, self.kernel_q_mat2))
        key_states_ker = self.ker_act(torch.einsum('bhsd,hde->bhse', key_states, self.kernel_k_mat1))
        key_states_ker = torch.abs(self.scalingD) * self.kernel_f(torch.einsum('bhsd,hde->bhse', key_states_ker, self.kernel_k_mat2))
        key_states_ker = key_states_ker + torch.einsum('bhsd,hde->bhse', key_states_ker, self.interaction_k) * self.scalingD2
        
        query_states_ker = query_states_ker.abs()
        key_states_ker = key_states_ker.abs()
        
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

        
        if self.attention_masks_next is not None:
            query_states_ker = query_states_ker.to(torch.float32)
            attn_weights = attn_weights.to(torch.float32)
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
            
            sparse_norms_lse = torch.logsumexp(attn_weights, -1, keepdim=True) #BHS1
            lr_norms = torch.matmul(query_states_ker, self.z) #BHSK,1HK1 -> BHS1
            norms_lse = torch.logaddexp(torch.log(torch.max(lr_norms, torch.tensor(1e-6))), sparse_norms_lse)
            lr_attn_out = torch.exp(torch.log(torch.max(query_states_ker, torch.tensor(1e-6))) - norms_lse) # scale queries #BHSK

            lr_attn_out = torch.matmul(lr_attn_out, self.H.to(query_states_ker.dtype)) #BHSK,1HKD -> BHSD S=1 or prompt
            sm_scale = torch.exp(sparse_norms_lse - norms_lse)

            query_states_ker = query_states_ker.to(hidden_states.dtype)
            attn_weights = attn_weights.to(hidden_states.dtype)
            sm_scale = sm_scale.to(hidden_states.dtype)
            lr_attn_out = lr_attn_out.to(hidden_states.dtype)
            del sparse_norms_lse, lr_norms, norms_lse
            

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
        
        # Accumulate attention scores
        if not self.previous_scores == None:
            current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
        
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum #(heads, k-tokens)
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
    
        if attn_tokens_all > self.cache_budget:
            # activate most recent k-cache
            if not self.recent_budget == 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                # activate historical best self.cache_budget - self.recent_budget tokens.
                # self.previous_scores # (k-Cache - 1)
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                if self.fix_heavy_to_initial_tokens:
                    attn_mask[..., :self.heavy_budget] = 1
                else:
                    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                    attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        prev_mask = self.attention_masks_next
        self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)
        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        if prev_mask is not None:
            eliminated_mask = prev_mask - self.attention_masks_next[..., :-1]
        else:
            eliminated_mask = 1 - self.attention_masks_next[..., :-1]
        
        # Update low-rank states
        z_update = (key_states_ker.transpose(2, 3) * eliminated_mask) #BHKs
        self.z = self.z + z_update.sum(dim=-1, keepdim=True)
        H_update = torch.matmul(z_update, value_states * eliminated_mask.transpose(2, 3)) #BHKD
        self.H = self.H + H_update
        
        attn_output = torch.matmul(attn_weights, value_states)
        if prev_mask is not None:
            attn_output = lr_attn_out + (attn_output * sm_scale)
            
        del prev_mask

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
        
        checkpoint[f'model.layers.{li}.self_attn.kernel_q_mat1'] = layer_loaded_data['kernel_q_mat1']
        checkpoint[f'model.layers.{li}.self_attn.kernel_k_mat1'] = layer_loaded_data['kernel_k_mat1']
        checkpoint[f'model.layers.{li}.self_attn.kernel_q_mat2'] = layer_loaded_data['kernel_q_mat2']
        checkpoint[f'model.layers.{li}.self_attn.kernel_k_mat2'] = layer_loaded_data['kernel_k_mat2']
        checkpoint[f'model.layers.{li}.self_attn.scalingD'] = layer_loaded_data['scalingD']
        checkpoint[f'model.layers.{li}.self_attn.interaction_k'] = layer_loaded_data['interaction_k']
        checkpoint[f'model.layers.{li}.self_attn.scalingD2'] = layer_loaded_data['scalingD2']
    model.load_state_dict(checkpoint)
    return model.to(device)



