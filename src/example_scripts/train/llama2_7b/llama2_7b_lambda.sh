save_dir=llama2_7b_lambda

python train_kernels.py \
    --save_dir ../checkpoints/$save_dir \
    --model_name llama2 \
    --model_size 0 \
    --sampling_batch_size 2 \
    --seqs_to_collect 512 \
    --half_precision \
    --heavy_ratio 0.001 \
    --recent_ratio 0.049 \
    --fix_heavy_to_initial_tokens \
    --ker_hid 512 \
    --ker_dim 8 \
    --lr 0.001 \
    --batch_size 2 \
    --epochs 40 \
    --device cuda:0