model_arch=llama2
model_size=0
task=wikitext
shots=0
device=0

# Full Cache
python eval_harness.py \
    --model_arch $model_arch \
    --model_size $model_size \
    --tasks $task \
    --num_fewshot $shots \
    --device cuda:$device

# H20 Baseline
python eval_harness.py \
    --model_arch $model_arch \
    --model_size $model_size \
    --enable_small_cache \
    --heavy_ratio 0.025 \
    --recent_ratio 0.025 \
    --tasks $task \
    --num_fewshot $shots \
    --device cuda:$device

# H20 Baseline+
python eval_harness.py \
    --model_arch $model_arch \
    --model_size $model_size \
    --enable_small_cache \
    --heavy_ratio 0.0255 \
    --recent_ratio 0.0255 \
    --tasks $task \
    --num_fewshot $shots \
    --device cuda:$device

# LESS
python eval_harness.py \
    --saved_model_name llama2_7b_h2o \
    --model_arch $model_arch \
    --model_size $model_size \
    --ker_dim 8 \
    --ker_hid 512 \
    --enable_small_cache \
    --heavy_ratio 0.025 \
    --recent_ratio 0.025 \
    --tasks $task \
    --num_fewshot $shots \
    --device cuda:$device