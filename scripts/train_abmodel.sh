module load anaconda3
source activate abmodel

export N_GPUS=4
export BASE_MODEL=path_to_pretrained_model
export RM=path_to_RM
export SAVE_DIR=path_to_save
export EXPERIMENT_NAME='mut_penalty'
export VLLM_ATTENTION_BACKEND=XFORMERS


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="verl/data/train.parquet" \
    data.val_files=""verl/data/val_10.parquet"" \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1000 \
    data.max_response_length=1000 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    reward_model.enable=True \
    reward_model.model.path=$RM \
    reward_model.micro_batch_size=16 \
    critic.optim.lr=5e-6 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=16 \
    trainer.critic_warmup=20 \
    trainer.logger=['tensorboard'] \
    trainer.project_name='verl_ab_generate' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=80 \
    trainer.test_freq=800 \
    trainer.default_local_dir=$SAVE_DIR \
    ++trainer.val_before_train=False \
    trainer.total_epochs=1 2>&1 | tee verl_demo_${EXPERIMENT_NAME}.log