python ./train/sft_attenuation.py \
    --base_model "your path" \
    --train_data_path='["your path"]' \
    --val_data_path='["your path"]' \
    --output_token_weights '[3.0, 1.0, 1.0]' \
    --train_on_inputs False \
    --batch_size 128 \
    --micro_batch_size 12 \
    --num_epochs 20 \
    --output_dir "save model path" \
    --k1 3 --m1 15 \
    --k2 3 --m2 15
    #--resume_from_checkpoint "/mnt/Qwen3-0.6B-addtoken"

# 