#!/bin/bash
eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
eval "$(conda shell.bash hook)"
source activate cogact


gpu_id=1

declare -a arr=(
  "/data/ckpt/cogact/pretrained/CogACT-Base/checkpoints/CogACT-Base.pt"
)

declare -a coke_can_options_arr=(
  "lr_switch=True"
  "upright=True"
  "laid_vertically=True"
)

EvalOverlay() {
  ckpt_path=$1
  env_name=$2
  scene_name=$3
  extra_args=$4

  eval_dir=$(dirname "$(dirname "${ckpt_path}")")/eval/$(basename "${ckpt_path}")
  mkdir -p "${eval_dir}"
  log_file="${eval_dir}/pick_coke_can_variant_agg.txt"

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
      --robot-init-x 0.35 0.35 1 \
      --robot-init-y 0.20 0.20 1 \
      --obj-init-x -0.35 -0.12 5 \
      --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      ${extra_args}

    echo "+###############################+"
    date
    echo "Done!"
    echo "+###############################+"
  } 2>&1 | tee -a "${log_file}"
}

# 1. Base setup
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
for ckpt_path in "${arr[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "--additional-env-build-kwargs ${coke_can_option}"
  done
done

# 2. Table textures
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=(
  "google_pick_coke_can_1_v4_alt_background"
  "google_pick_coke_can_1_v4_alt_background_2"
)
for ckpt_path in "${arr[@]}"; do
  for scene in "${scene_arr[@]}"; do
    for coke_can_option in "${coke_can_options_arr[@]}"; do
      EvalOverlay "$ckpt_path" "$env_name" "$scene" "--additional-env-build-kwargs ${coke_can_option}"
    done
  done
done

# 3. Distractors
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4
for ckpt_path in "${arr[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    # 无 distractor_config 参数的调用
    EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "--additional-env-build-kwargs ${coke_can_option}"
    # 添加 distractor_config=more 的调用
    EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "--additional-env-build-kwargs ${coke_can_option} distractor_config=more"
  done
done

# 4. Backgrounds
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=(
  "google_pick_coke_can_1_v4_alt_background"
  "google_pick_coke_can_1_v4_alt_background_2"
)
for ckpt_path in "${arr[@]}"; do
  for scene in "${scene_arr[@]}"; do
    for coke_can_option in "${coke_can_options_arr[@]}"; do
      EvalOverlay "$ckpt_path" "$env_name" "$scene" "--additional-env-build-kwargs ${coke_can_option}"
    done
  done
done

# 5. Lightings
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
for ckpt_path in "${arr[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "--additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True"
    EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "--additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True"
  done
done

# 6. Camera orientations
declare -a env_arr=(
  "GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0"
  "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0"
)
scene_name=google_pick_coke_can_1_v4
for ckpt_path in "${arr[@]}"; do
  for env in "${env_arr[@]}"; do
    for coke_can_option in "${coke_can_options_arr[@]}"; do
      EvalOverlay "$ckpt_path" "$env" "$scene_name" "--additional-env-build-kwargs ${coke_can_option}"
    done
  done
done
