#!/bin/bash
set -x
ENGINE=${1:-vllm}

# Model configuration
MODEL_PATH="$HOME/models/GLM-4.1V-9B"

# Data configuration
train_path="$HOME/data/geo3k/geo3k/train.parquet"
test_path="$HOME/data/geo3k/test.parquet"

# Experiment configuration
project_name="glm4-1v-9b"
experiment_name="$(date +%Y%m%d)-geo3k-test"
export TENSORBOARD_DIR=$project_name/$experiment_name

mkdir -p logs

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.custom_cls.path=recipe/glm4v/rl_dataset.py \
    data.custom_cls.name=RLHFDataset \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager='prime' \
    reward_model.launch_reward_fn_async=False \
    algorithm.use_kl_in_reward=False \
    ray_init.num_cpus=16 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 $@ 2>&1 | tee logs/${project_name}_${experiment_name}.log