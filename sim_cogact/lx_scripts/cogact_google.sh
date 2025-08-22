#!/bin/bash
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
eval "$(conda shell.bash hook)"
source activate cogact

#ckpt_path=/data/ckpt/cogact/pretrained/CogACT-Base/checkpoints/CogACT-Base.pt
#ckpt_path=/data/ckpt/cogact/log/reproduce/fractal_ditb--image_aug/checkpoints/step-030000-epoch-02-loss=0.0426.pt
#action_model_type="DiT-B"
ckpt_path=/data/ckpt/cogact/pretrained/CogACT-Large/checkpoints/CogACT-Large.pt
action_model_type="DiT-L"

policy_setup="google_robot"
cfg_dirs=(
  "sim_cogact/sim_inference_cfgs/google_drawer_variant_agg"
  "sim_cogact/sim_inference_cfgs/google_drawer_visual_matching"
  "sim_cogact/sim_inference_cfgs/google_move_near_variant_agg"
  "sim_cogact/sim_inference_cfgs/google_move_near_visual_matching"
  "sim_cogact/sim_inference_cfgs/google_pick_coke_can_variant_agg"
  "sim_cogact/sim_inference_cfgs/google_pick_coke_can_visual_matching"
  "sim_cogact/sim_inference_cfgs/google_put_in_drawer_variant_agg"
  "sim_cogact/sim_inference_cfgs/google_put_in_drawer_visual_matching"
)

for i in "${!cfg_dirs[@]}"; do
    cfg_dir=${cfg_dirs[$i]}
    gpu_id=$i
    echo "Start task：cfg_dir=${cfg_dir} on GPU ${gpu_id}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python sim_cogact/main_inference_fast.py \
        --cfgs_dir "${cfg_dir}" \
        --ckpt_path "${ckpt_path}" \
        --policy_setup "${policy_setup}" \
        --action_model_type "${action_model_type}" &
done

wait
echo "🎉 All done!"
