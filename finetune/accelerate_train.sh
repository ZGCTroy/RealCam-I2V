export TOKENIZERS_PARALLELISM=false

WORKSPACE=$(dirname "$0")
cd $WORKSPACE

ACCELERATE_CONFIG_FILE=${WORKSPACE}/accelerate_config.yaml

PRETRAINED_MODEL_DIR=$(dirname "$0")/pretrained

DATA_ROOT=$(dirname "$0")/data/RealCam-Vid

SPLIT=train


CHECKPOINT_DIR=$(dirname "$0")/checkpoints
EXPERIMENT_NAME=RealCam-I2V
SUB_EXPERIMENT_NAME=CogVideoX1.5-5B-ControlNetXs
LOG_DIR=${CHECKPOINT_DIR}/${EXPERIMENT_NAME}/${SUB_EXPERIMENT_NAME}
mkdir -p ${LOG_DIR}
export WANDB_DIR=${LOG_DIR}

# Model Configuration
MODEL_ARGS=(
    --model_path ${PRETRAINED_MODEL_DIR}/CogVideoX1.5-5B-I2V
    --model_name "cogvideox1.5-i2v"
    --model_type "i2v"
    --training_type "controlnetxs"
    --time_sampling_type "truncated_normal"
    --time_sampling_mean 0.8
    --time_sampling_type 0.075
    --keep_aspect_ratio
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir $LOG_DIR
    --report_to "wandb"
    --tracker_name $EXPERIMENT_NAME
    --sub_tracker_name $SUB_EXPERIMENT_NAME
)

# Training Configuration
TRAIN_ARGS=(
    --train_steps 50000
    --batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate 4e-5
    --weight_decay 1e-4
    --mixed_precision "bf16"  # ["no", "fp16"]
    --gradient_checkpointing
    --enable_slicing
    --enable_tiling
    --seed 42
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 4
    --pin_memory
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100
    --checkpointing_limit 100
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation
    --validation_dir ${CHECKPOINT_DIR}
    --validation_steps 100
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 8
)

# extract video latents of 81x256x448 ; "768//3 x 1360//3 "
DATA_ARGS=(
    --data_root ${DATA_ROOT}
    --cache_root $(dirname "$0")/data/cache
    --metadata_path RealCam-Vid_new_${SPLIT}.npz
    --enable_align_factor
)

# distribution args for multi-node
DIST_ARGS=(
    --config_file $ACCELERATE_CONFIG_FILE
    --num_machines $HOST_NUM
    --num_processes $NODE_NUM
    --machine_rank $INDEX
    --main_process_ip $CHIEF_IP
    --main_process_port 29500
)

accelerate launch "${DIST_ARGS[@]}" train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \ 
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    --train_resolution "81x768x1360"  \
    --precompute

# Optional for landscape/portrait joint training
# accelerate launch "${DIST_ARGS[@]}" train.py \
#     "${MODEL_ARGS[@]}" \
#     "${OUTPUT_ARGS[@]}" \
#     "${DATA_ARGS[@]}" \
#     "${TRAIN_ARGS[@]}" \
#     "${SYSTEM_ARGS[@]}" \
#     "${CHECKPOINT_ARGS[@]}" \
#     "${VALIDATION_ARGS[@]}" \
#     --train_resolution "81x1360x768"  \
#     --precompute

accelerate launch ${DIST_ARGS[@]} train.py \
    ${MODEL_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${SYSTEM_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${VALIDATION_ARGS[@]} \
    --train_resolution "81x768x1360"  \
    # --allow_switch_hw
