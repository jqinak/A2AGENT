# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535
export HF_HOME=/project/peilab/qjl/tmp
export TMPDIR=/project/peilab/qjl/tmp
export RAY_TMPDIR=/project/peilab/qjl/tmp

cd /project/peilab/qjl/A2AGENT

PROJECT_DIR="$(pwd)"
DATA_DIR=/project/peilab/qjl/CODE/DATA
DATASET=a2agent_rl

PORT=16259

lsof -i :$PORT
kill -9 $(lsof -t -i:$PORT)

MODELPATH=$PROJECT_DIR/a2agent/model/Qwen3-VL-4B-Instruct

# python -m a2agent.model.download_model --repo-id "" --token --local-dir $MODELPATH

# python -m a2agent.data.a2agent_preprocess_rl


nohup python -m a2agent.server.async_sandbox_server ALLOWED_PORT=$PORT > $PROJECT_DIR/a2agent/server/server.log 2>&1 &

NAME=qwen3_4b_a2agent_test0
PROJECT=a2agent_async_rl



TOOL_CONFIG_PATH=$PROJECT_DIR/a2agent/tools/config/a2agent_tool_config.yaml


python3 -m a2agent.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=16384 \
    data.max_response_length=20000 \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=512 \
    data.truncation=left \
    data.image_key=images \
    data.video_key=videos \
    custom_reward_function.path=pkg://a2agent.reward.reward_a2po \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=multimodalcode \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=8 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.model.path="$MODELPATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    data.train_files="$PROJECT_DIR/a2agent/data/$DATASET/train.parquet" \
    data.val_files="$PROJECT_DIR/a2agent/data/$DATASET/test.parquet" \
    trainer.total_epochs=3 \
    "$@" 2>&1 | tee "$PROJECT_DIR/scripts/$NAME.log"

lsof -i :$PORT
kill -9 $(lsof -t -i:$PORT)