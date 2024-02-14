# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/main

import argparse
import json
import logging

from lm_eval import tasks, evaluator, utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from inference.parallel.llama import convert_kvcache_llama_sparse, convert_kvcache_llama_less
from inference.parallel.falcon import convert_kvcache_falcon_sparse, convert_kvcache_falcon_less


logging.getLogger("openai").setLevel(logging.WARNING)



ENABLE_SPARSE_FUNCTIONS = {
    "llama2": convert_kvcache_llama_sparse,
    "falcon": convert_kvcache_falcon_sparse
}

ENABLE_LESS_FUNCTIONS = {
    "llama2": convert_kvcache_llama_less,
    "falcon": convert_kvcache_falcon_less,
}


models_sizes_dict = {
    'llama2': ['7b', '13b', '70b'],
    'falcon': ['7b', '40b'],
}

hugging_name_dict = {
    'llama2': lambda x: f'meta-llama/Llama-2-{x}-hf',
    'falcon': lambda x: f'tiiuae/falcon-{x}'
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    # LM Harness Args
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    # Basic Configs
    parser.add_argument("--saved_model_name", type=str, default='')
    parser.add_argument('--model_arch', type=str, default='llama2')
    parser.add_argument("--model_size", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    # Evaluation Sparse Policy (does not need to the same as the trained policy)
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument('--fix_heavy_to_initial_tokens', action='store_true')
    
    # Kernels
    parser.add_argument("--ker_dim", type=int, default=8)
    parser.add_argument("--ker_hid", type=int, default=512)
    
    # Misc
    parser.add_argument("--sample_checkpoint", type=int, default=0) # Start eval at sample i
    parser.add_argument("--device", type=str, default='cpu')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    
    if len(task_names) > 1:
        raise NotImplementedError
    
    model_name = hugging_name_dict[args.model_arch](models_sizes_dict[args.model_arch][args.model_size])
    # shots = args.shots
    print(model_name)

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    
    if args.enable_small_cache:
        print('Enable Small Cache Size')
        saved_model_name = args.saved_model_name
        config.fix_heavy_to_initial_tokens = args.fix_heavy_to_initial_tokens
        config.heavy_count = int(args.heavy_ratio * config.max_position_embeddings)
        config.recent_count = int(args.recent_ratio * config.max_position_embeddings)
        config.kernel_hidden_size = args.ker_dim
        config.ker_hid = args.ker_hid
        
        if saved_model_name == '':
            model = ENABLE_SPARSE_FUNCTIONS[args.model_arch](model, config)
        else:
            path_func = lambda li: f'../checkpoints/{saved_model_name}/layer_{li}.pth'
            model = ENABLE_LESS_FUNCTIONS[args.model_arch](model, config, path_func)
        
    model = model.half()
    model = model.eval().to(args.device)


    # assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=model,
        # model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        sample_checkpoint=args.sample_checkpoint,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        tokenizer=tokenizer,
    )
    
    print(evaluator.make_table(results))

if __name__ == "__main__":
    main()

