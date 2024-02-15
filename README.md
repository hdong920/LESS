# LESS

This repository contains the implementation of LESS (**L**ow-rank **E**mbedding **S**idekick with **S**parse policy), presented in ["Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference"](https://arxiv.org/abs/2402.09398)


Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang (Atlas) Wang, Yuejie Chi, Beidi Chen


### Abstract

Many computational factors limit broader deployment of large language models. In this paper, we focus on a memory bottleneck imposed by the key-value (KV) cache, a computational shortcut that requires storing previous KV pairs during decoding. While existing KV cache methods approach this problem by pruning or evicting large swaths of relatively less important KV pairs to dramatically reduce the memory footprint of the cache, they can have limited success in tasks that require recollecting a majority of previous tokens. To alleviate this issue, we propose LESS, a simple integration of a (nearly free) constant sized cache with eviction-based cache methods, such that all tokens can be queried at later decoding steps. Its ability to retain information throughout time shows merit on a variety of tasks where we demonstrate LESS can help reduce the performance gap from caching everything, sometimes even matching it, all while being efficient.


### Usage

Example scripts to train and validate using different models and methods are included in `src/example_scripts`. Training checkpoints can be found in `checkpoints`.

#### Setup

Clone this repository, and then set up the conda environment as follows:

```bash
conda env create -f less.yml
conda activate less
cd src
```

#### Training

Example training scripts can be found in `src/example_scripts/train`. For instance, to train LESS for Llama 2 7B with 5% H2O, run

```bash
sh example_scripts/train/llama2_7b/llama2_7b_h2o.sh
```

Similarly, to train LESS for Falcon 7B with 10% H2O, run:

```bash
sh example_scripts/train/falcon_7b/falcon_7b_h2o.sh
```

These scripts will train LESS on a single device sequentially. As desribed in our paper, LESS is trained separately and independently for each attention layer, so training can be easily parallelized on multiple devices by distributing a model's layers on different devices. The arguments `--from_layer` and `--to_layer` for training can be useful for this. 


#### Evaluation

This section provides details on evaluating the performance of LESS and its benchmarks. Example training scripts can be found in `src/example_scripts/eval`. Following the structure of experiments in the [H2O repository](https://github.com/FMInference/H2O/tree/main), we make the distinction between generation and non-generation (parallel) tasks. More details can be found in the paper. The implementations of LESS and baselines will differ for both. 

To evaluate your trained LESS for Llama 2 7B on WikiText (parallel), run 

```bash
sh example_scripts/eval/llama2_7b/wikitext.sh
```

and similarly for other lm-harness tasks. For summarization on CNN/DailyMail (generative), run

```bash
sh example_scripts/eval/llama2_7b/cnn_dailymail.sh
```
and similarly for other summarization datasets.

<!-- We have also provided trained LESS kernels for Llama 2 7B with 5% H2O in `checkpoints/llama2_7b_less_5_h2o`. Simply replace the original argument `--saved_model_name` with "llama2_7b_less_5_h2o" in `example_scripts/eval/llama2_7b/wikitext.sh` and `example_scripts/eval/llama2_7b/cnn_dailymail.sh` before running the above evaluation examples. -->


### Citation

If you found this repository helpful in your work, please cite our [paper](https://arxiv.org/abs/2402.09398):

    @misc{dong2024less,
      title={Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference}, 
      author={Harry Dong and Xinyu Yang and Zhenyu Zhang and Zhangyang Wang and Yuejie Chi and Beidi Chen},
      year={2024},
      eprint={2402.09398},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
