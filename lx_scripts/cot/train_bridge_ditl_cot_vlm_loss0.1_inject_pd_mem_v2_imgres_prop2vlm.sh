#!/bin/bash
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
eval "$(conda shell.bash hook)"
source activate cogact_pd

pretrained_ckpt='/data/ckpt/cogact/pretrained/CogACT-Large/checkpoints/CogACT-Large.pt'
action_model_type='DiT-L'

data_mix='bridge'
data_root_dir='/mnt/robopt/openx_256'

n_gpu=8
bs=2
traj_group_size=16
shuffle_buffer_size=51_200

save_interval=5000
dp_step=4
future_action_window_size=15

image_aug=True
run_root_dir='/data/ckpt/cogact/log/cot'
run_id='bridge_ditl_cot_vlm_loss0.1_inject_pd_mem_v2_group16_imgres_prop2vlm'

is_resume=True
resume_step=5000
resume_epoch=0

torchrun --standalone --nnodes 1 --nproc-per-node ${n_gpu} scripts/train.py \
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
  --wandb_project 'cogact' \
  --wandb_entity 'shihao-thu' \
  --hf_token 'xxxx' \
  --use_cot True \
  --cot_file_path /data/ckpt/ecot/embodied_features_bridge/embodied_features_bridge.json \
  --vlm_loss_weight 0.1 \
  --lang_inject True \
  --traj_group_size ${traj_group_size} \
  --vla.shuffle_buffer_size ${shuffle_buffer_size} \
  --use_memory_bank True \
  --memory_bank_version v2 \
  --use_img_res True \
  --img_res_share_vision_encoder True \
  --proprio_to_vlm True
