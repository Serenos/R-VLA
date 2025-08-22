gpu_id=0
export PYTHONPATH=/home/lixiang10/workspace/R-VLA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"
policy_model=cogact

ckpt_path="/home/lixiang10/workspace/R-VLA/CogACT/cot_chekpoint/bridge_ditb_cot_v6.2-cot_vlm_loss0.1_inject-hicotv2-group32_from-pretrained--image_aug/checkpoints/step-010000-epoch-00-loss=0.1173.pt"

lang_inject=hicot_v2
cot_frozen_step=6
cot_memory_expire=6
cot_generate_sample=6
cot_allow_prefix=-1
action_model_type=DiT-B
eval_note="eval_cot_v62_fromDiTL_coarse2fine_20k_frozen${cot_frozen_step}_cotmemoryv2_expire${cot_memory_expire}_dosample${cot_generate_sample}_inject${lang_inject}_control"


scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

eval_dir=$(dirname $(dirname ${ckpt_path}))/${eval_note}/$(basename ${ckpt_path})
mkdir -p ${eval_dir}
env_name=StackGreenCubeOnYellowCubeBakedTexInScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/R-VLA/sim_cogact/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type ${action_model_type} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version v6 \
  --cot_frozen_step ${cot_frozen_step} \
  --lang_inject ${lang_inject} \
  --use_cot_memory \
  --cot_memory_expire ${cot_memory_expire} \
  --cot_generate_sample ${cot_generate_sample} \
  --cot_allow_prefix ${cot_allow_prefix} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file}


env_name=PutCarrotOnPlateInScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/R-VLA/sim_cogact/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type ${action_model_type} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version v6 \
  --cot_frozen_step ${cot_frozen_step} \
  --lang_inject ${lang_inject} \
  --use_cot_memory \
  --cot_memory_expire ${cot_memory_expire} \
  --cot_generate_sample ${cot_generate_sample} \
  --cot_allow_prefix ${cot_allow_prefix} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file}

env_name=PutSpoonOnTableClothInScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/R-VLA/sim_cogact/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type ${action_model_type} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version v6 \
  --cot_frozen_step ${cot_frozen_step} \
  --lang_inject ${lang_inject} \
  --use_cot_memory \
  --cot_memory_expire ${cot_memory_expire} \
  --cot_generate_sample ${cot_generate_sample} \
  --cot_allow_prefix ${cot_allow_prefix} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file}


scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path="./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
robot_init_x=0.127
robot_init_y=0.06

env_name=PutEggplantInBasketScene-v0
log_file=${eval_dir}/${env_name}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python /home/lixiang10/workspace/R-VLA/sim_cogact/main_inference.py  --policy-model ${policy_model} --ckpt-path ${ckpt_path} --cogact-action-model-type ${action_model_type} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --use_cot \
  --cot_version v6 \
  --cot_frozen_step ${cot_frozen_step} \
  --lang_inject ${lang_inject} \
  --use_cot_memory \
  --cot_memory_expire ${cot_memory_expire} \
  --cot_generate_sample ${cot_generate_sample} \
  --cot_allow_prefix ${cot_allow_prefix} \
  --eval_note ${eval_note} \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${log_file}
