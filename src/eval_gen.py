#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
import argparse
import logging

import numpy as np
import torch
import json
import tqdm 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from rouge import Rouge

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
models_sizes_dict = {
    'llama2': ['7b', '13b', '70b'],
    'falcon': ['7b', '40b'],
}

hugging_name_dict = {
    'llama2': lambda x: f'meta-llama/Llama-2-{x}-hf', 
    'falcon': lambda x: f'tiiuae/falcon-{x}',
}


def load_modules(use_low_rank):
    from inference.generative.llama_gen import convert_kvcache_llama_sparse, LlamaAttentionSparse, convert_kvcache_llama_less, LlamaAttentionLESS
    from inference.generative.falcon_gen import convert_kvcache_falcon_sparse, FalconAttentionSparse, convert_kvcache_falcon_less, FalconAttentionLESS

    if not use_low_rank:
        ENABLE_FUNCTIONS = {
            "llama2": convert_kvcache_llama_sparse,
            "falcon": convert_kvcache_falcon_sparse
        }
        TARGET_MODULE = {
            "llama2": LlamaAttentionSparse,
            'falcon': FalconAttentionSparse
        }
    else:
        ENABLE_FUNCTIONS = {
            "llama2": convert_kvcache_llama_less,
            "falcon": convert_kvcache_falcon_less
        }
        TARGET_MODULE = {
            "llama2": LlamaAttentionLESS,
            'falcon': FalconAttentionLESS
        }
    return ENABLE_FUNCTIONS, TARGET_MODULE


def main():
    parser = argparse.ArgumentParser()
    
    # Basic Configs
    parser.add_argument("--saved_model_name", type=str, default='')
    parser.add_argument('--model_arch', type=str, default='llama2')
    parser.add_argument('--model_size', type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    # Evaluation Sparse Policy (does not need to the same as the trained policy)
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument('--fix_heavy_to_initial_tokens', action='store_true')
    
    # Kernels
    parser.add_argument("--ker_dim", type=int, default=8)
    parser.add_argument("--ker_hid", type=int, default=512)
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='cnn')
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    
    # Misc
    parser.add_argument("--sample_checkpoint", type=int, default=0) # Start eval at sample i
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    set_seed(args)

    shots = args.shots
    if args.dataset == 'cnn':
        input_paths = [f'../data/cnn_data/cnn_dailymail_{shots}shot.jsonl']
    elif args.dataset == 'xsum':
        input_paths = [f'../data/xsum_data/xsum_{shots}shot.jsonl']
    elif args.dataset == 'multinews':
        input_paths = [f'../data/multinews_data/multinews_{shots}shot_1.jsonl', f'../data/multinews_data/multinews_{shots}shot_2.jsonl'] 
    else:
        raise NotImplementedError
    
    sample_checkpoint = args.sample_checkpoint
    # checkpoint_rate = args.checkpoint_rate

    model_size_name = models_sizes_dict[args.model_arch][args.model_size]

    config = AutoConfig.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(hugging_name_dict[args.model_arch](model_size_name))

    
    if args.enable_small_cache:
        print('Enable Small Cache Size')
        saved_model_name = args.saved_model_name
        config.fix_heavy_to_initial_tokens = args.fix_heavy_to_initial_tokens
        config.heavy_count = int(args.heavy_ratio * config.max_position_embeddings)
        config.recent_count = int(args.recent_ratio * config.max_position_embeddings)
        config.kernel_hidden_size = args.ker_dim
        config.ker_hid = args.ker_hid
        
        ENABLE_FUNCTIONS, TARGET_MODULE = load_modules(saved_model_name != '')
        
        if saved_model_name == '':
            model = ENABLE_FUNCTIONS[args.model_arch](model, config)
        else:
            path_func = lambda li: f'../checkpoints/{saved_model_name}/layer_{li}.pth'
            model = ENABLE_FUNCTIONS[args.model_arch](model, config, path_func)
    else:
        ENABLE_FUNCTIONS, TARGET_MODULE = load_modules(False)

    
    model.eval().to(args.device)
    logger.info(args)

    if args.max_length == -1:
        args.max_length = config.max_position_embeddings

    requests = []
    for input_path in input_paths:
         with open(input_path, 'r') as f:
             for line in f:
                 if line.strip() != '':
                     requests.append(json.loads(line))

    print(len(requests))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()

    seq_lens = []
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    skipped=0
    with torch.no_grad():
        for i, request in enumerate(tqdm.tqdm(requests)):
            if i < sample_checkpoint:
                continue

            stop = ['###']
            temperature = 0.3
            prompt = request['article']
            label = request['summary_gt']
            max_tokens = args.max_tokens
            result = {}

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            if len(input_ids[0]) > args.max_length-max_tokens:
                skipped+=1
                print('skipped', skipped)

            else:
                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=max_tokens + len(input_ids[0]),
                    temperature=temperature,
                    top_k=args.k,
                    top_p=1,
                    do_sample=True,
                    num_return_sequences=1,
                    return_dict_in_generate=True, output_scores=True,
                )

                for name, m in model.named_modules():
                    if isinstance(m, TARGET_MODULE[args.model_arch]):
                        m._reset_masks()

                tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
                logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
                top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

                generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
                generate_text = generate_text[: generate_text.find(stop[0])]

                scores = rouge.get_scores(generate_text, label)[0]
                seq_lens.append(len(input_ids[0]))
                rouge1_score_list.append(scores['rouge-1']['f'])
                rouge2_score_list.append(scores['rouge-2']['f'])
                rougel_score_list.append(scores['rouge-l']['f'])

                result['result'] = {
                    "choices": [
                        {
                            "text": generate_text,
                            "logprobs": {
                                "tokens": tokens, 
                                "token_logprobs": logprobs, 
                                "top_logprobs": top_logprobs, 
                                "text_offset": []
                            }, 
                            "finish_reason": "length"
                        }
                    ], 
                    "request_time": {
                        "batch_time": 0, 
                        "batch_size": 1}
                }
                
                results.append(result)
                print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
    
    print("FINAL RESULTS")
    print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

if __name__ == "__main__":
    main()

