import os
import sys
import json
import argparse
import numpy as np
from loguru import logger
from datetime import datetime


from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from sim_cogact import CogACTInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgs_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--policy_setup", type=str, required=True, help="google_robot or widowx_bridge")
    parser.add_argument("--action_model_type", type=str, required=True, help="DiT-B or DiT-L")
    # parser.add_argument("--traj_group_size", type=int, default=1, help="group size for trajectory sampling")
    # parser.add_argument("--use_memory_bank", action="store_true", help="whether to use memory bank")
    
    parser.add_argument("--traj_group_size", type=int, default=1,)
    parser.add_argument("--use_memory_bank", action="store_true", default=False,)
    parser.add_argument("--use_img_res", action="store_true", default=False,)
    parser.add_argument("--img_res_share_vision_encoder", action="store_true", default=False,)
    parser.add_argument("--load_proprio", action="store_true", default=False,)
    parser.add_argument("--proprio_to_vlm", action="store_true", default=False,)
    parser.add_argument("--memory_bank_version", type=str, default="v1", choices=["v1", "v2"],)
    parser.add_argument("--lang_action_out", action="store_true", default=False,)
    parser.add_argument("--use_cot", action="store_true", default=False,)
    parser.add_argument("--cot_version", type=str, default="None", choices=["None", "v1", "v2", "v3", "v3.2", "v4", "v5", "v6", "v6_sample", "vqa"],)
    parser.add_argument("--cot_frozen_step", type=int, default=12, help="Number of steps to freeze the prompt in CoT reasoning.")
    parser.add_argument("--cot_memory_expire", type=int, default=6, help=".")
    parser.add_argument("--eval_note", type=str, default="None",)
    parser.add_argument("--lang_inject", type=str, default="no")
    parser.add_argument("--use_moe", action="store_true", default=False)
    parser.add_argument("--use_cot_memory", action="store_true", default=False)
    parser.add_argument("--use_cot_trigger", action="store_true", default=False)
    parser.add_argument("--cot_generate_sample", type=int, default=-1, help="Whether to generate with sampling for CoT reasoning.")
    parser.add_argument("--cot_allow_prefix", type=int, default=-1, help="Whether to allow prefix in CoT reasoning.")
    parser.add_argument("--action_ensemble", action="store_true", default=True)
    parser.add_argument("--action_chunk", action="store_true", default=False)
    args = parser.parse_args()
    # os.environ['XDG_RUNTIME_DIR'] = '/usr/lib/'
    os.environ['DISPLAY'] = ""
    # os.environ['__GLX_VENDOR_LIBRARY_NAME'] = "nvidia"
    os.environ['VK_ICD_FILENAMES'] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

    # # prevent a single jax process from taking up all the GPU memory
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # gpus = tf.config.list_physical_devices("GPU")
    # if len(gpus) > 0:
    #     # prevent a single tf process from taking up all the GPU memory
    #     tf.config.set_logical_device_configuration(
    #         gpus[0],
    #         [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
    #     )

    logging_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'eval')
    os.makedirs(logging_dir, exist_ok=True)
    log_file = os.path.join(logging_dir, os.path.basename(args.ckpt_path), os.path.basename(args.cfgs_dir) + '.txt')

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, level="INFO", rotation="1 MB", encoding="utf-8")

    logger.info("+" * 40)
    logger.info(f"Loading model from {args.ckpt_path}")
    # model = CogACTInference(
    #     saved_model_path=args.ckpt_path,
    #     policy_setup=args.policy_setup,
    #     action_scale=1.0,
    #     action_model_type=args.action_model_type,
    #     cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
    #     traj_group_size=args.traj_group_size,
    #     use_memory_bank=args.use_memory_bank,
    #     use_img_res=args.use_img_res,
    #     img_res_share_vision_encoder=args.img_res_share_vision_encoder,
    # )
    model = CogACTInference(
        saved_model_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
        action_model_type=args.cogact_action_model_type,
        cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
        traj_group_size=args.traj_group_size,
        use_memory_bank=args.use_memory_bank,
        use_img_res=args.use_img_res,
        img_res_share_vision_encoder=args.img_res_share_vision_encoder,
        load_proprio=args.load_proprio,
        proprio_to_vlm=args.proprio_to_vlm,
        memory_bank_version=args.memory_bank_version,
        lang_action_out=args.lang_action_out,
        use_cot=args.use_cot,
        action_ensemble=args.action_ensemble,
        action_chunk=args.action_chunk,
        cot_version=args.cot_version,
        lang_inject=args.lang_inject,
        use_cot_memory=args.use_cot_memory,
        cot_memory_expire=args.cot_memory_expire,
        cot_frozen_step=args.cot_frozen_step,
        use_cot_trigger=args.use_cot_trigger,
        cot_generate_sample=args.cot_generate_sample,
        cot_allow_prefix=args.cot_allow_prefix,
    )
    logger.info("Model loaded successfully")
    logger.info("+" * 40)

    cfg_files = [os.path.join(args.cfgs_dir, f) for f in os.listdir(args.cfgs_dir) if f.endswith(".json")]

    all_avg_success = []
    for cfg_file in cfg_files:
        logger.info("+" * 40)
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Processing cfg: {cfg_file}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            args_list = json.load(f)

        sys.argv = [sys.argv[0]] + args_list
        eval_args = get_args()
        eval_args.logging_dir = logging_dir

        logger.info("\nArguments:\n" + ",\n".join(
            f'  {json.dumps(k)}: {json.dumps(v, separators=(",", ": "), ensure_ascii=False, default=lambda o: o.tolist())}'
            for k, v in vars(eval_args).items()
        ))

        success_arr = maniskill2_evaluator(model, eval_args)
        avg_success = float(np.mean(success_arr))
        all_avg_success.append(avg_success)

        logger.info("Success array: " + str(success_arr))
        logger.info(f"Average success: {avg_success}")
        logger.info("+" * 40)

    logger.info("+" * 40)
    overall_avg = float(np.mean(all_avg_success)) if all_avg_success else 0.0
    logger.info("Overall average success for all cfgs: " + str(overall_avg))
    logger.info("+" * 40)


if __name__ == "__main__":
    main()
