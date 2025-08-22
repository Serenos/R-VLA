gpu_id=0
export PYTHONPATH=/home/lixiang10/workspace/CogACT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"
policy_model=cogact

ckpt_path="/home/lixiang10/workspace/CogACT/CogACT/cot_chekpoint/bridge_ditb_cot_v3.2_plan+position_vlm_loss0.1--image_aug/checkpoints/step-020000-epoch-02-loss=0.0470.pt"
eval_note="eval_cot_v3.2_gripper_frozen12"

cot_version="v3.2"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

eval_dir=$(dirname $(dirname ${ckpt_path}))/${eval_note}/$(basename ${ckpt_path})
mkdir -p ${eval_dir}
env_name=StackGreenCubeOnYellowCubeBakedTexInScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/CogACT/sim_cogact/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type DiT-B \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version ${cot_version} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file};


env_name=PutCarrotOnPlateInScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/CogACT/sim_cogact/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type DiT-B \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version ${cot_version} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file};

env_name=PutSpoonOnTableClothInScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/CogACT/sim_cogact/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type DiT-B \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version ${cot_version} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file};


scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path="./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
robot_init_x=0.127
robot_init_y=0.06

env_name=PutEggplantInBasketScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/CogACT/sim_cogact/main_inference.py  --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type DiT-B \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version ${cot_version} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file};
