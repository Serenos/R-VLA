#!/bin/bash
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
eval "$(conda shell.bash hook)"
source activate cogact

#nohup sudo Xvfb :66 -screen 0 640x360x24 &
#export DISPLAY=:66
#sudo startxfce4 &

policy_model=cogact

ckpt_paths=(
/data/ckpt/cogact/pretrained/CogACT-Base/checkpoints/CogACT-Base.pt
)

gpu_ids=(0 1 2 3)
action_model_type=DiT-B
#gpu_ids=(4 5 6 7)
#action_model_type=DiT-L

run_inference() {
    local env_name=$1
    local scene_name=$2
    local robot=$3
    local rgb_overlay_path=$4
    local robot_init_x=$5
    local robot_init_y=$6
    local gpu_id=$7
    local log_file=${eval_dir}/${env_name}.txt

    CUDA_VISIBLE_DEVICES=${gpu_id} python sim_cogact/main_inference.py \
      --policy-model ${policy_model} \
      --cogact-action-model-type ${action_model_type} \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name ${env_name} \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file} &
}


env_names=(
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0"
    "PutCarrotOnPlateInScene-v0"
    "PutSpoonOnTableClothInScene-v0"
    "PutEggplantInBasketScene-v0"
)
scene_names=(
    "bridge_table_1_v1"
    "bridge_table_1_v1"
    "bridge_table_1_v1"
    "bridge_table_1_v2"
)
robots=(
    "widowx"
    "widowx"
    "widowx"
    "widowx_sink_camera_setup"
)
rgb_overlay_paths=(
    "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
    "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
    "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
    "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
)
SimplerEnv_PATH="./third_libs/SimplerEnv"
for i in "${!rgb_overlay_paths[@]}"; do
    rgb_overlay_paths[$i]="$SimplerEnv_PATH/${rgb_overlay_paths[$i]}"
done

robot_init_xs=(0.147 0.147 0.147 0.127)
robot_init_ys=(0.028 0.028 0.028 0.06)


for ckpt_path in "${ckpt_paths[@]}"; do
  eval_dir=$(dirname $(dirname ${ckpt_path}))/eval/$(basename ${ckpt_path})
  mkdir -p ${eval_dir}

  num_tasks=${#env_names[@]}
  for i in $(seq 0 $((num_tasks - 1))); do
      gpu_id=${gpu_ids[$((i % ${#gpu_ids[@]}))]}
      run_inference "${env_names[i]}" "${scene_names[i]}" "${robots[i]}" "${rgb_overlay_paths[i]}" "${robot_init_xs[i]}" "${robot_init_ys[i]}" "${gpu_id}"
  done
  wait
  echo "Done: ${ckpt_path}"
done

wait
echo "🎉 All done!"
