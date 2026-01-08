#!/usr/bin/env bash
# python examples/data_preprocess/pokemon.py
set -xeuo pipefail
export NCCL_TOPO_FILE=''
# export VERL_LOGGING_LEVEL='INFO'
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

# TRAIN_FILES=/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/a2agent/data/test_sft/train.parquet
# TEST_FILES=/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/a2agent/data/test_sft/test.parquet

DATASET=spacer_c_map_singleturn
TRAIN_FILES=/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/a2agent/data/$DATASET/train.parquet
TEST_FILES=/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/a2agent/data/$DATASET/test.parquet
# TRAIN_FILES=a2agent/data/spacer_c_map_singleturn/train.parquet
# TEST_FILES=a2agent/data/spacer_c_map_singleturn/test.parquet
backend=${BACKEND:-fsdp}

project_name=verl_sft_test

RESUME_MODE=auto
MODEL_ID=/nvme/data-pool1/qjl/MODEL/Qwen3-VL-4B-Instruct
# MODEL_ID=${HDFS_ROOT}/model/Qwen3-VL-30B-A3B-Instruct

# SP_SIZE=${SP_SIZE:-2}
# TP_SIZE=${TP_SIZE:-2}
# PP_SIZE=${PP_SIZE:-2}
# VPP_SIZE=${VPP_SIZE:-null}
# CP_SIZE=${CP_SIZE:-1}
PAD_MODE=${PAD_MODE:-no_padding}



FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.strategy=fsdp2 \
    engine.fsdp_size=-1"

    # engine.ulysses_sequence_parallel_size=${SP_SIZE} \

ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
echo "Using fsdp engine"


exp_name=qwen3_vl_4b_$DATASET


CKPT_HOME=/home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/checkpoints/${exp_name}
mkdir -p "${CKPT_HOME}"

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS:-8} \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TEST_FILES}" \
    data.train_batch_size=96 \
    data.max_length=2048 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=65536 \
    model.path=$MODEL_ID \
    model.use_remove_padding=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=2 \
    checkpoint.save_contents=[model,optimizer,extra] 2>&1 | tee /home/jincai_guo/ICML2025_JIE/QJL/A2AGENT/scripts/sft_spacer_c_map.log