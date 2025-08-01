"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, GroupPaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, GroupedRLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    traj_group_size: int = 1,
    use_optim_group_sample: bool = False,
    load_depth: bool = False,
    load_proprio: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,  # Concatenated `past_action_window_size-1' actions and the current action for the input
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    lang_action_out: bool = False,
    use_cot: bool = False,
    cot_version: str = None,
    cot_file_path: str = None,
) -> Tuple[Dataset, ActionTokenizer, Union[PaddedCollatorForActionPrediction, GroupPaddedCollatorForActionPrediction]]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        lang_action_out=lang_action_out,
        use_cot=use_cot,
        cot_version=cot_version,
        cot_dropout_prob=0.0,
        print_prompt_limit=20,
        co_training_prob=0.8,
    )

    # Build RLDS Iterable Dataset
    if traj_group_size > 1:
        assert episodic is False, "Grouped dataset is not supported for episodic dataset"

        collator = GroupPaddedCollatorForActionPrediction(
            tokenizer.model_max_length,
            tokenizer.pad_token_id,
            padding_side=padding_side,
            load_proprio=load_proprio,
        )

        dataset = GroupedRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            traj_group_size=traj_group_size,
            use_optim_group_sample=use_optim_group_sample,
            load_depth=load_depth,
            load_proprio=load_proprio,
            use_cot=use_cot,
            cot_file_path=cot_file_path,
        )
    else:
        collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length,
            tokenizer.pad_token_id,
            padding_side=padding_side,
            load_proprio=load_proprio,
        )

        cls = RLDSDataset if not episodic else EpisodicRLDSDataset
        dataset = cls(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            load_depth=load_depth,
            load_proprio=load_proprio,
            use_cot=use_cot,
            cot_file_path=cot_file_path,
        )

    return dataset, action_tokenizer, collator
