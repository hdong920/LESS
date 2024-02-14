model_arch=falcon
model_size=0
dataset=cnn
shots=3
sample_num=1000
device=0

# Full Cache
python eval_gen.py \
    --model_arch $model_arch \
    --model_size $model_size \
    --dataset $dataset \
    --shots $shots \
    --sample_num $sample_num \
    --device cuda:$device

# H20 Baseline
python eval_gen.py \
    --model_arch $model_arch \
    --model_size $model_size \
    --dataset $dataset \
    --shots $shots \
    --sample_num $sample_num \
    --enable_small_cache \
    --heavy_ratio 0.05 \
    --recent_ratio 0.05 \
    --device cuda:$device

# H20 Baseline+
python eval_gen.py \
    --model_arch $model_arch \
    --model_size $model_size \
    --dataset $dataset \
    --shots $shots \
    --sample_num $sample_num \
    --enable_small_cache \
    --heavy_ratio 0.051 \
    --recent_ratio 0.051 \
    --device cuda:$device

# LESS
python eval_gen.py \
    --saved_model_name falcon_7b_h2o \
    --model_arch $model_arch \
    --model_size $model_size \
    --dataset $dataset \
    --shots $shots \
    --sample_num $sample_num \
    --ker_dim 8 \
    --ker_hid 512 \
    --enable_small_cache \
    --heavy_ratio 0.05 \
    --recent_ratio 0.05 \
    --device cuda:$device
