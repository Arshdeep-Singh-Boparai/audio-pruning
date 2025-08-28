eval "$('~/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate panns

cd ./audioset_tagging_cnn/

WORKSPACE="./workspaces/audioset_tagging"   # Default argument.
echo $WORKSPACE

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
    --workspace=$WORKSPACE \
    --data_type='full_train' \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='QResNet38_pruned' \ #or QCnn14_pruned, specify model name here, change p, path_to_sorted_indexes (L1,GM,OP)
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --resume_iteration=0 \
    --early_stop=1000000 \
    --cuda




