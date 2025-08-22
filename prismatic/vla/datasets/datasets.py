"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
import random

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.util.cot_utils import CotTag, abbreviate_tag, get_cot_tags_list
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset, make_interleaved_episodic_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def cot_dropout(cot: str, dropout_prob: float) -> Tuple[str, str]:
    """Dropout cot tokens with probability `dropout_prob`."""
    if len(cot) == 0:
        return cot, ""

    cot_parts = cot.split("@")
    tags = [(cot_parts[i], cot_parts[i + 1]) for i in range(0, len(cot_parts), 2)]

    subset = np.random.rand(len(tags)) > dropout_prob

    subset_string = (
        "[" + ", ".join([abbreviate_tag(tag) for (tag, _), is_taken in zip(tags, subset) if is_taken]) + "]"
    )  # abbreviation

    # excluded_tags = [CotTag.MOVE_KEYFRAME.value, CotTag.SUBTASK_KEYFRAME.value, CotTag.VISION_KEYFRAME.value]
    excluded_tags = [
        CotTag.MOVE_KEYFRAME.value,
        CotTag.SUBTASK_KEYFRAME.value,
        CotTag.VISION_KEYFRAME.value,
        CotTag.GRIPPER_POSITION.value,
    ]

    if "EXCLUDE_TAGS" in os.environ:
        excluded_tags = os.environ["EXCLUDE_TAGS"].split(",")

    return (
        " ".join(
            [f"{tag[0]} {tag[1]}" for tag, is_taken in zip(tags, subset) if (is_taken and tag[0] not in excluded_tags)]
        ),
        subset_string,
    )


def cot_dropout_v3(cot: str, dropout_prob: float, excluded_tags) -> Tuple[str, str]:
    """Dropout cot tokens with probability `dropout_prob`."""
    if len(cot) == 0:
        return ""

    cot_parts = cot.split("@")
    tags = [(cot_parts[i], cot_parts[i + 1]) for i in range(0, len(cot_parts), 2)]

    subset = np.random.rand(len(tags)) > dropout_prob

    if "EXCLUDE_TAGS" in os.environ:
        excluded_tags = os.environ["EXCLUDE_TAGS"].split(",")

    return " ".join(
        [f"{tag[0]} {tag[1]}" for tag, is_taken in zip(tags, subset) if (is_taken and tag[0] not in excluded_tags)]
    )


def parse_cot(cot_str):
    if len(cot_str) == 0:
        return {}
    parts = cot_str.split("@")
    return {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}


def select_cot_fields_by_timestep(cot_dict, timestep, traj_length):
    # 早期
    if timestep <= 5:
        tags = [CotTag.TASK.value, CotTag.PLAN.value]
        return select_ordered_random_cot_fields(cot_dict, tags)
    # 中期
    elif 5 < timestep < 10:  # 你可以根据实际情况调整"中期"范围
        tags = [CotTag.VISIBLE_OBJECTS.value, CotTag.SUBTASK_REASONING.value, CotTag.MOVE_REASONING.value]
        return select_ordered_random_cot_fields(cot_dict, tags)
    # 后期
    else:
        tags = [CotTag.SUBTASK.value, CotTag.MOVE.value, CotTag.GRIPPER_POSITION.value]
        return select_ordered_random_cot_fields(cot_dict, tags)


def select_cot_fields_by_keytimestep(cot_dict, timestep, traj_length):

    if CotTag.SUBTASK_KEYFRAME.value in cot_dict:
        subtask_keyframe = cot_dict[CotTag.SUBTASK_KEYFRAME.value]
    else:
        subtask_keyframe = 0
    if CotTag.MOVE_KEYFRAME.value in cot_dict:
        move_keyframe = cot_dict[CotTag.MOVE_KEYFRAME.value]
    else:
        move_keyframe = 0
    if CotTag.VISION_KEYFRAME.value in cot_dict:
        vision_keyframe = cot_dict[CotTag.VISION_KEYFRAME.value]
    else:
        vision_keyframe = 0
    # 早期
    if timestep <= 0.1 * traj_length:  # (0, 5)
        tags = [CotTag.TASK.value, CotTag.PLAN.value]
        if vision_keyframe == "1":
            tags += [CotTag.VISIBLE_OBJECTS.value]
        if subtask_keyframe == "1":
            tags += [CotTag.SUBTASK_REASONING.value, CotTag.SUBTASK.value]
        if move_keyframe == "1":
            tags += [CotTag.MOVE_REASONING.value, CotTag.MOVE.value, CotTag.GRIPPER_POSITION.value]
    # 中期
    elif 0.1 * traj_length < timestep < 0.2 * traj_length:  # (5, 10)
        tags = []
        if vision_keyframe == "1":
            tags += [CotTag.VISIBLE_OBJECTS.value]
        if subtask_keyframe == "1":
            tags += [CotTag.SUBTASK_REASONING.value, CotTag.SUBTASK.value]
        if move_keyframe == "1":
            tags += [CotTag.MOVE_REASONING.value, CotTag.MOVE.value]
    # 后期
    else:
        tags = []
        if subtask_keyframe == "1":
            tags += [CotTag.SUBTASK.value]
        if move_keyframe == "1":
            tags += [CotTag.MOVE.value]
        if subtask_keyframe == "1" or move_keyframe == "1":
            tags += [CotTag.GRIPPER_POSITION.value]
        tags = [CotTag.SUBTASK.value, CotTag.MOVE.value, CotTag.GRIPPER_POSITION.value]
    return select_cot_dropout_fields(cot_dict, tags)


def gaussian_weight(mu, sigma, t):
    return np.exp(-((t - mu) ** 2) / (2 * sigma**2))


def select_soft_cot_fields_by_timestep(cot_dict, timestep, traj_length, top_k=3):
    d = max(traj_length - 8, 0)
    subtask_timestep = d * 0.25 + 8
    move_timestep = d * 0.5 + 8
    gripper_timestep = d * 0.75 + 8
    mu_sigma_table = {
        CotTag.TASK.value: (4, 2),
        CotTag.PLAN.value: (4, 2),
        CotTag.VISIBLE_OBJECTS.value: (8, 2),
        CotTag.SUBTASK_REASONING.value: (8, 2),
        CotTag.MOVE_REASONING.value: (8, 2),
        CotTag.SUBTASK.value: (subtask_timestep, 5),
        CotTag.MOVE.value: (move_timestep, 5),
        CotTag.GRIPPER_POSITION.value: (gripper_timestep, 2),
    }
    available_tags = [tag for tag in mu_sigma_table.keys() if tag in cot_dict]
    if not available_tags:
        return ""
    weights = {}
    for tag, (mu, sigma) in mu_sigma_table.items():
        weights[tag] = gaussian_weight(mu, sigma, timestep)
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    random_topk = random.randint(0, top_k)
    if random_topk == 0:
        return ""
    topk_tags = set([tag for tag, _ in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:random_topk]])
    selected = [tag for tag in available_tags if tag in topk_tags]
    return " ".join([f"{tag} {cot_dict[tag]}" for tag in selected])


def select_random_cot_fields(cot_dict, tags):
    available_tags = [tag for tag in tags if tag in cot_dict]
    if not available_tags:
        return ""
    k = random.randint(1, len(available_tags))
    chosen_tags = random.sample(available_tags, k)
    return " ".join([f"{tag} {cot_dict[tag]}" for tag in chosen_tags])


def select_ordered_random_cot_fields(cot_dict, tags):
    available_tags = [tag for tag in tags if tag in cot_dict]
    if not available_tags:
        return ""
    all_combinations = []
    n = len(available_tags)
    for i in range(1, 2**n):
        combo = [available_tags[j] for j in range(n) if (i >> j) & 1]
        all_combinations.append(combo)
    chosen_tags = random.choice(all_combinations)
    return " ".join([f"{tag} {cot_dict[tag]}" for tag in chosen_tags])


def select_cot_dropout_fields(cot_dict, tags, dropout_prob=0.3):
    available_tags = [tag for tag in tags if tag in cot_dict]
    if not available_tags:
        return ""
    chosen = np.random.rand(len(available_tags)) > dropout_prob
    chosen_tags = [available_tags[i] for i in range(len(available_tags)) if chosen[i]]
    return " ".join([f"{tag} {cot_dict[tag]}" for tag in chosen_tags])


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    lang_action_out: bool = False
    use_cot: bool = False
    cot_version: str = "v1"
    cot_dropout_prob: float = 0.0
    print_prompt_limit: int = 10
    co_training_prob: float = 0.8

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        if self.use_cot:
            cot, _ = cot_dropout(rlds_batch["cot"].decode(), dropout_prob=self.cot_dropout_prob)
        else:
            cot = ""

        if self.cot_version == "v1":
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"{cot}"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v2":
            assert self.use_cot or self.lang_action_out
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"</s>{cot}"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v3" or self.cot_version == "v3.2":
            current_action = rlds_batch["action"][0]
            action_str = self.action_tokenizer(current_action)
            random_use_cot = random.random() < self.co_training_prob
            if random_use_cot:
                cot_fields = get_cot_tags_list()
                choose_cot_fields = [CotTag.SUBTASK.value, CotTag.GRIPPER_POSITION.value]
                field = random.choice(choose_cot_fields)
                excluded_tags = [f for f in cot_fields if f != field]
                cot_v3 = cot_dropout_v3(
                    rlds_batch["cot"].decode(), dropout_prob=self.cot_dropout_prob, excluded_tags=excluded_tags
                )
                # v3.0
                # conversation = [
                #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
                #     {"from": "gpt", "value": f"{cot_v3} {CotTag.ACTION.value} {action_str}"},
                # ]
                # v3.1
                # conversation = [
                #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
                #     {"from": "gpt", "value": f"{cot_v3}"},
                # ]
                # v3.2
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"</s>{cot_v3}"},
                ]
            else:
                cot_v3 = ""
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"</s>{cot_v3}"},
                ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v4":
            random_use_cot = random.random() < self.co_training_prob
            if random_use_cot:
                cot_fields = get_cot_tags_list()
                choose_cot_fields = [CotTag.SUBTASK.value, CotTag.GRIPPER_POSITION.value]
                field = random.choice(choose_cot_fields)
                excluded_tags = [f for f in cot_fields if f != field]
                cot_v4 = cot_dropout_v3(
                    rlds_batch["cot"].decode(), dropout_prob=self.cot_dropout_prob, excluded_tags=excluded_tags
                )

                conversation = [
                    {
                        "from": "human",
                        "value": f"What action should the robot take to {lang} based on {field.lower()[:-1]}?",
                    },
                    {"from": "gpt", "value": f"</s>{cot_v4}"},
                ]
            else:
                cot_v4 = ""
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"</s>{cot_v4}"},
                ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v5":
            random_use_cot = random.random() < self.co_training_prob
            if random_use_cot:
                timestep = int(rlds_batch["observation"]["timestep"][0])
                traj_length = int(rlds_batch["observation"]["traj_length"][0])
                cot_dict = parse_cot(rlds_batch["cot"].decode())
                cot_v5 = select_cot_fields_by_timestep(cot_dict, timestep, traj_length)
            else:
                cot_v5 = ""

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"</s>{cot_v5}"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v5.2":
            random_use_cot = random.random() < self.co_training_prob
            if random_use_cot:
                timestep = int(rlds_batch["observation"]["timestep"][0])
                traj_length = int(rlds_batch["observation"]["traj_length"][0])
                cot_dict = parse_cot(rlds_batch["cot"].decode())
                cot_v5 = select_cot_fields_by_timestep(cot_dict, timestep, traj_length)
                cot_progress = f"TASK PROGRESS: {timestep / traj_length:.1f}."
                if random.random() < 0.3:
                    cot_value = f"</s>{cot_progress} {cot_v5}"
                else:
                    cot_value = f"</s>{cot_v5}"
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": cot_value},
                ]
            else:
                cot_v5 = ""
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"</s>{cot_v5}"},
                ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v5.3":
            random_use_cot = random.random() < self.co_training_prob
            if random_use_cot:
                timestep = int(rlds_batch["observation"]["timestep"][0])
                traj_length = int(rlds_batch["observation"]["traj_length"][0])
                cot_dict = parse_cot(rlds_batch["cot"].decode())
                # cot_v5 = select_cot_fields_by_timestep(cot_dict, timestep, traj_length)
                cot_v5 = select_soft_cot_fields_by_timestep(cot_dict, timestep, traj_length)
                # cot_progress = f'TASK PROGRESS: {timestep / traj_length:.2f}'
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"</s>{cot_v5}"},
                ]
            else:
                cot_v5 = ""
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"</s>{cot_v5}"},
                ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v6":
            random_use_cot = random.random() < self.co_training_prob
            if random_use_cot:
                timestep = int(rlds_batch["observation"]["timestep"][0])
                traj_length = int(rlds_batch["observation"]["traj_length"][0])
                cot_dict = parse_cot(rlds_batch["cot"].decode())
                cot_v6 = select_cot_fields_by_keytimestep(cot_dict, timestep, traj_length)
            else:
                cot_v6 = ""

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"</s>{cot_v6}"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v6.1":
            random_use_cot = random.random() < self.co_training_prob
            use_full_cot = random_use_cot and (random.random() < 0.2)

            if random_use_cot:
                timestep = int(rlds_batch["observation"]["timestep"][0])
                traj_length = int(rlds_batch["observation"]["traj_length"][0])
                cot_dict = parse_cot(rlds_batch["cot"].decode())
                if use_full_cot:
                    cot_v6, _ = cot_dropout(rlds_batch["cot"].decode(), dropout_prob=0.2)
                else:
                    cot_v6 = select_cot_fields_by_keytimestep(cot_dict, timestep, traj_length)
            else:
                cot_v6 = ""

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"</s>{cot_v6}"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "v6.2":
            # random_use_cot = random.random() < self.co_training_prob
            use_full_cot = random.random() < 0.2  # 新增：20%使用完整 CoT

            timestep = int(rlds_batch["observation"]["timestep"][0])
            traj_length = int(rlds_batch["observation"]["traj_length"][0])
            cot_dict = parse_cot(rlds_batch["cot"].decode())
            if use_full_cot:
                cot_v6, _ = cot_dropout(rlds_batch["cot"].decode(), dropout_prob=0.2)
            else:
                cot_v6 = select_cot_fields_by_keytimestep(cot_dict, timestep, traj_length)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"</s>{cot_v6}"},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        elif self.cot_version == "vqa":
            random_use_cot = random.random() < self.co_training_prob
            cot_dict = parse_cot(rlds_batch["cot"].decode())
            current_action = rlds_batch["action"][0]
            action_str = self.action_tokenizer(current_action)
            # 逐条列出问题和答案
            if random_use_cot:
                conversation = []
                if CotTag.TASK.value in cot_dict:
                    conversation.append(
                        [
                            {"from": "human", "value": "Explain the task?"},
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.TASK.value]}"},
                        ]
                    )
                if CotTag.PLAN.value in cot_dict and CotTag.TASK.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": f"What's the plan for the task '{cot_dict[CotTag.TASK.value].lower()}'?",
                            },
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.PLAN.value]}"},
                        ]
                    )
                if CotTag.VISIBLE_OBJECTS.value in cot_dict and CotTag.TASK.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": (
                                    f"Where is the object related to the task '{cot_dict[CotTag.TASK.value].lower()}'?"
                                ),
                            },
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.VISIBLE_OBJECTS.value]}"},
                        ]
                    )
                if CotTag.MOVE.value in cot_dict and CotTag.MOVE_REASONING.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": (
                                    f"Why should the robot perform the movement '{cot_dict[CotTag.MOVE.value].lower()}'?"
                                ),
                            },
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.MOVE_REASONING.value]}"},
                        ]
                    )
                if CotTag.SUBTASK.value in cot_dict and CotTag.SUBTASK_REASONING.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": (
                                    f"Why should the robot perform the subtask '{cot_dict[CotTag.SUBTASK.value].lower()}'?"
                                ),
                            },
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.SUBTASK_REASONING.value]}"},
                        ]
                    )
                if CotTag.SUBTASK.value in cot_dict and CotTag.MOVE.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": (
                                    f"What's the movement to perform the subtask '{cot_dict[CotTag.SUBTASK.value].lower()}'?"
                                ),
                            },
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.MOVE.value]}"},
                        ]
                    )
                if CotTag.MOVE.value in cot_dict and CotTag.GRIPPER_POSITION.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": (
                                    f"What's the gripper's position to perform the movement '{cot_dict[CotTag.MOVE.value].lower()}'?"
                                ),
                            },
                            {"from": "gpt", "value": f"</s>{cot_dict[CotTag.GRIPPER_POSITION.value]}"},
                        ]
                    )
                if CotTag.GRIPPER_POSITION.value in cot_dict and CotTag.ACTION.value in cot_dict:
                    conversation.append(
                        [
                            {
                                "from": "human",
                                "value": (
                                    f"What's the action to follow the gripper position '{cot_dict[CotTag.GRIPPER_POSITION.value].lower()}'?"
                                ),
                            },
                            {"from": "gpt", "value": f"</s>{action_str}"},
                        ]
                    )
                if conversation:
                    conversation = random.choice(conversation)
                else:
                    conversation = [
                        {"from": "human", "value": f"What action should the robot take to {lang}?"},
                        {"from": "gpt", "value": "</s>"},
                    ]
            else:
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": "</s>"},
                ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        # Mask prompt tokens
        if self.cot_version is not None:
            labels[: int(torch.where(input_ids == 2)[0][0])] = IGNORE_INDEX
        # if self.cot_version == 'v3':
        #     base_prompt = f'In: What action should the robot take to {lang}?\nOut: '
        #     prompt_input_ids = self.base_tokenizer(base_prompt, add_special_tokens=False).input_ids
        #     prompt_token_len = len(prompt_input_ids)
        #     labels[:prompt_token_len] = IGNORE_INDEX

        if self.print_prompt_limit > 0:
            p = prompt_builder.get_prompt()
            print("--------------------------------")
            print("Prompt:", p)
            print("Labels:", labels)
            print("--------------------------------")
            self.print_prompt_limit -= 1

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        if "proprio" in rlds_batch["observation"]:
            proprio = torch.tensor(rlds_batch["observation"]["proprio"], dtype=action.dtype)
        else:
            proprio = None

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=action,
            action_masks=action_mask,
            proprio=proprio,
        )


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        future_action_window_size: int = 0,
        past_action_window_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
        load_all_data_for_training: bool = True,
        load_depth=False,
        load_proprio=False,
        use_cot: bool = False,
        cot_file_path: str = None,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=load_depth,
            load_proprio=load_proprio,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=past_action_window_size + 1,                                    # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                                        # Skip trajectories without language labels
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            load_all_data_for_training=load_all_data_for_training,
            use_cot=use_cot,
            cot_file_path=cot_file_path,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            # load_all_data_for_training=rlds_config["load_all_data_for_training"],
        )

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class GroupedRLDSDataset(RLDSDataset):
    def __init__(self, *args, traj_group_size: int = 16, use_optim_group_sample: bool = False, **kwargs):
        self.traj_group_size = traj_group_size
        self.use_optim_group_sample = use_optim_group_sample
        super().__init__(*args, **kwargs)

    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            traj_group_size=self.traj_group_size,
            use_optim_group_sample=self.use_optim_group_sample,
        )

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            T = rlds_batch["action"].shape[0]
            if self.use_optim_group_sample:
                indices = range(T)
            elif T < self.traj_group_size:
                # If trajectory length is less than traj_group_size, pad with last frame
                indices = np.arange(T)
                pad_count = self.traj_group_size - T
                pad_indices = np.full((pad_count,), T - 1)
                indices = np.concatenate([indices, pad_indices])
            else:
                # Randomly sample `traj_group_size` frames from the trajectory (without replacement),
                # and ensure they are sorted in temporal order by using np.sort
                indices = np.random.choice(T, size=self.traj_group_size, replace=False)
                indices = np.sort(indices)

            out = [self.batch_transform(tree_map(lambda x: x[i], rlds_batch)) for i in indices]  # noqa: B023

            yield out
