save_dir=falcon_7b_h2o

python train_kernels.py \
    --save_dir ../checkpoints/$save_dir \
    --model_name falcon \
    --model_size 0 \
    --sampling_batch_size 2 \
    --seqs_to_collect 1024 \
    --half_precision \
    --heavy_ratio 0.05 \
    --recent_ratio 0.05 \
    --ker_hid 512 \
    --ker_dim 8 \
    --lr 0.001 \
    --batch_size 2 \
    --epochs 40 \
    --device cuda:0