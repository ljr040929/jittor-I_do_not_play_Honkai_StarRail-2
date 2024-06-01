#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="../../A"
OUTPUT_DIR_PREFIX="style/style_objects/style_"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=500
LEARNING_RATE=0.00015
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=100
MAX_TRAIN_STEPS=600
SEED=0
GPU_COUNT=1
MAX_NUM=13

LORA_RANK=32

for ((folder_number = 13; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        PROMPT_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)"
        STYLE_PROMPT=$(jq '.style' "$PROMPT_DIR/prompt_new.json")
        CLASS_PROMPT=$(jq '.class' "$PROMPT_DIR/prompt_new.json")
        CAPTION_JSON=$(jq '.caption' "$PROMPT_DIR/prompt_new.json")
        echo "$CAPTION_JSON" | jq -c '.[]' | while read line; do
            CUDA_VISIBLE_DEVICES=$gpu_id
            CAPTION=$(echo $line | sed 's/\"//g')
            OBJECT_DIR="\"${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/object/${CAPTION}\""
            OUTPUT_DIR="\"${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)/${CAPTION}\""
            
            COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_object.py \
                --pretrained_model_name_or_path=$MODEL_NAME \
                --instance_data_dir=$OBJECT_DIR \
                --output_dir=$OUTPUT_DIR \
                --style_instance_prompt=$STYLE_PROMPT \
                --object_instance_prompt="\"$CAPTION"\" \
                --resolution=$RESOLUTION \
                --train_batch_size=$TRAIN_BATCH_SIZE \
                --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
                --learning_rate=$LEARNING_RATE \
                --lr_scheduler=$LR_SCHEDULER \
                --lr_warmup_steps=$LR_WARMUP_STEPS \
                --max_train_steps=$MAX_TRAIN_STEPS \
                --seed=$SEED \
                --rank=$LORA_RANK"
            eval $COMMAND &
            sleep 3
            wait
        done
    done
done