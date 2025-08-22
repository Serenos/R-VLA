#!/bin/bash
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
eval "$(conda shell.bash hook)"
source activate cogact


gpu_id=2

declare -a arr=(
"/data/ckpt/cogact/pretrained/CogACT-Base/checkpoints/CogACT-Base.pt"

#"/data/ckpt/cogact/log/reproduce/fractal_ditb--image_aug/checkpoints/step-010000-epoch-00-loss=0.0480.pt"
#"/data/ckpt/cogact/log/reproduce/fractal_ditb--image_aug/checkpoints/step-020000-epoch-01-loss=0.0422.pt"
#"/data/ckpt/cogact/log/reproduce/fractal_ditb--image_aug/checkpoints/step-030000-epoch-02-loss=0.0426.pt"

)

# coke_can_options：设置罐子姿态（例如，水平翻转、直立、垂直放置）
declare -a coke_can_options_arr=(
  "lr_switch=True"
  "upright=True"
  "laid_vertically=True"
)

declare -a urdf_version_arr=(
  None
  "recolor_tabletop_visual_matching_1"
  "recolor_tabletop_visual_matching_2"
  "recolor_cabinet_visual_matching_1"
)

env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png

EvalOverlay() {
  ckpt_path=$1
  env_name=$2
  scene_name=$3
  rgb_overlay_path=$4
  coke_can_option=$5
  urdf_version=$6
  
  eval_dir=$(dirname "$(dirname "${ckpt_path}")")/eval/$(basename "${ckpt_path}")
  mkdir -p "${eval_dir}"
  log_file="${eval_dir}/pick_coke_can_visual_matching.txt"

  {
    echo "+###############################+"
    date
    for i in "$@"; do
      echo "| $i"
    done
    echo "+###############################+"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python sim_cogact/main_inference.py \
      --policy-model cogact \
      --ckpt-path "${ckpt_path}" \
      --robot google_robot_static \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 80 \
      --env-name "${env_name}" \
      --scene-name "${scene_name}" \
      --rgb-overlay-path "${rgb_overlay_path}" \
      --robot-init-x 0.35 0.35 1 \
      --robot-init-y 0.20 0.20 1 \
      --obj-init-x -0.35 -0.12 5 \
      --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version}
    
    echo "+###############################+"
    date
    echo "Done!"
    echo "+###############################+"
  } 2>&1 | tee -a "${log_file}"
}

for ckpt_path in "${arr[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    for urdf_version in "${urdf_version_arr[@]}"; do
      EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "$rgb_overlay_path" "$coke_can_option" "$urdf_version"
    done
  done
done
