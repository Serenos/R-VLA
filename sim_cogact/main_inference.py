import os
import numpy as np
import tensorflow as tf
import argparse
import sys

# from ..third_libs.SimplerEnv.simpler_env.evaluation.argparse import get_args
# from ..third_libs.SimplerEnv.simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from sim_cogact import CogACTInference

from prismatic.util import set_global_seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--traj_group_size", type=int, default=1,)
    extra_parser.add_argument("--use_memory_bank", action="store_true", default=False,)
    extra_parser.add_argument("--use_img_res", action="store_true", default=False,)
    extra_parser.add_argument("--img_res_share_vision_encoder", action="store_true", default=False,)
    extra_parser.add_argument("--load_proprio", action="store_true", default=False,)
    extra_parser.add_argument("--proprio_to_vlm", action="store_true", default=False,)
    extra_parser.add_argument("--memory_bank_version", type=str, default="v1", choices=["v1", "v2"],)
    extra_parser.add_argument("--lang_action_out", action="store_true", default=False,)
    extra_parser.add_argument("--use_cot", action="store_true", default=False,)
    extra_parser.add_argument("--cot_version", type=str, default="None", choices=["None", "v1", "v2", "v3", "v3.2", "v4", "v5", "v6", "v6_sample", "vqa"],)
    extra_parser.add_argument("--cot_frozen_step", type=int, default=12, help="Number of steps to freeze the prompt in CoT reasoning.")
    extra_parser.add_argument("--cot_memory_expire", type=int, default=6, help=".")
    extra_parser.add_argument("--eval_note", type=str, default="None",)
    extra_parser.add_argument("--lang_inject", type=str, default="v1")
    extra_parser.add_argument("--use_moe", action="store_true", default=False)
    extra_parser.add_argument("--use_cot_memory", action="store_true", default=False)
    extra_parser.add_argument("--use_cot_trigger", action="store_true", default=False)
    extra_parser.add_argument("--cot_generate_sample", type=int, default=-1, help="Whether to generate with sampling for CoT reasoning.")
    extra_parser.add_argument("--cot_allow_prefix", type=int, default=-1, help="Whether to allow prefix in CoT reasoning.")
    extra_parser.add_argument("--action_ensemble", action="store_true", default=True)
    extra_parser.add_argument("--action_chunk", action="store_true", default=False)

    extra_args, remaining_args = extra_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args

    args = get_args()
    _ = set_global_seed(42, get_worker_init_fn=False)

    for key, value in vars(extra_args).items():
        setattr(args, key, value)

    args.logging_dir = os.path.join(os.path.dirname(
        os.path.dirname(args.ckpt_path)), args.eval_note)  # 'eval

    # os.environ["SVULKAN2_HEADLESS"] = "1"

    os.environ['DISPLAY'] = ''

    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=args.tf_memory_limit)],
        )

    if args.policy_model == "cogact":
        assert args.ckpt_path is not None
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
    else:
        raise NotImplementedError()

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
