module load anaconda3
source activate abmodel
bsz=4
lr=5e-5
model="mistral"

config_path=path_to_config
data_path=path_to_data
model_ckpt=path_to_model

output_dir=
logging_dir=
acc_config="ab/accelerate/accelerate_config.yaml"
train_log="train.log"

export PERCENT_DATA=1.0


accelerate launch --main_process_port 0 --config-file $acc_config train_rm.py \
    --config_path $config_path \
    --model_name_or_path $model_ckpt \
    --data_path $data_path \
    --do_train \
    --output_dir $output_dir \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 3e-5 \
    --lr_scheduler_type constant_with_warmup \
    --per_device_train_batch_size $bsz \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate $lr \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_steps 5000 \
    --max_steps 500000 \
    --logging_steps 10 \
    --logging_dir $logging_dir \
    --save_steps 5000 \
    --bf16 \
    --report_to tensorboard \
    --seed 42 \
    >$train_log 2>&1