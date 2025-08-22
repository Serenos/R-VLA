#!/bin/bash
# eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
# eval "$(conda shell.bash hook)"
# source activate cogact
export PYTHONPATH=/home/lixiang10/workspace/CogACT:$PYTHONPATH

pretrained_ckpt='/data/lixiang10/CogACT/CogACT-Base/checkpoints/CogACT-Base.pt'
#pretrained_ckpt='/data/lixiang10/CogACT/runs/debug--image_aug/checkpoints/step-025000-epoch-02-loss\=0.0264.pt'
# action_model_type='DiT-L'
action_model_type='DiT-B'

data_mix='bridge'
data_root_dir='/mnt/robotics-pretrain/openx_256/openx_256/'

n_gpu=8
bs=32
save_interval=10000
dp_step=4
future_action_window_size=15

image_aug=True
run_root_dir='/data/lixiang10/CogACT/cot_chekpoint'
run_id='bridge_ditb_stage1_cot_vlm_loss1'

is_resume=False
resume_step=0
resume_epoch=0

$(which python) -m torch.distributed.run \
  --standalone --nnodes 1 --nproc_per_node ${n_gpu} \
  scripts/train.py \
  --pretrained_checkpoint ${pretrained_ckpt} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${data_mix} \
  --vla.expected_world_size ${n_gpu} \
  --vla.per_device_batch_size ${bs} \
  --vla.global_batch_size $((n_gpu * bs)) \
  --vla.learning_rate 2e-5 \
  --data_root_dir ${data_root_dir} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug ${image_aug} \
  --save_interval ${save_interval} \
  --repeated_diffusion_steps ${dp_step} \
  --future_action_window_size ${future_action_window_size} \
  --action_model_type ${action_model_type} \
  --is_resume ${is_resume} \
  --resume_step ${resume_step} \
  --resume_epoch ${resume_epoch} \
  --wandb_project cogact-bridge \
  --wandb_entity lixiang_thu-tsinghua-university \
  --hf_token 'xxxx' \
  --use_cot True \
  --cot_version stage1 \
  --cot_file_path /data/lixiang10/embodiedCoT/embodied_features_bridge.json \
  --vlm_loss_weight 1 \
  --lang_inject False \
  --use_optim_group_sample True
