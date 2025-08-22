#!/bin/bash

export PYTHONPATH="/home/lixiang10/workspace/CogACT"
export CUDA_VISIBLE_DEVICES="0"

CKPT_PATH="/home/lixiang10/workspace/CogACT/CogACT/cot_chekpoint/bridge_ditb_cot_v3_vlm_loss0.1_inject--image_aug/checkpoints/step-020000-epoch-02-loss=0.0791.pt"
EVAL_NOTE="eval_cot_None_dropout_inject_frozen0"

declare -a ENV_NAMES=(
  "StackGreenCubeOnYellowCubeBakedTexInScene-v0"
  "PutCarrotOnPlateInScene-v0"
  "PutSpoonOnTableClothInScene-v0"
  "PutEggplantInBasketScene-v0"
)

declare -a SCENE_NAMES=(
  "bridge_table_1_v1"
  "bridge_table_1_v1"
  "bridge_table_1_v1"
  "bridge_table_1_v2"
)

declare -a ROBOTS=(
  "widowx"
  "widowx"
  "widowx"
  "widowx_sink_camera_setup"
)

declare -a RGB_OVERLAY_PATHS=(
  "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
  "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
  "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
  "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
)

# 只执行第一个任务
i=3
eval_dir="./eval_logs/${EVAL_NOTE}"
mkdir -p ${eval_dir}
log_file=${eval_dir}/${ENV_NAMES[$i]}.txt

python /home/lixiang10/workspace/CogACT/sim_cogact/main_inference.py \
  --policy-model cogact \
  --cogact-action-model-type DiT-B \
  --ckpt-path "$CKPT_PATH" \
  --robot "${ROBOTS[$i]}" \
  --policy-setup widowx_bridge \
  --control-freq 5 \
  --sim-freq 500 \
  --max-episode-steps 120 \
  --env-name "${ENV_NAMES[$i]}" \
  --scene-name "${SCENE_NAMES[$i]}" \
  --rgb-overlay-path "${RGB_OVERLAY_PATHS[$i]}" \
  --robot-init-x 0.147 0.147 1 \
  --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode \
  --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version None \
  --lang_inject \
  --eval_note ${EVAL_NOTE} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file}