import enum


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    SUBTASK_KEYFRAME = "SUBTASK KEYFRAME:"
    MOVE_KEYFRAME = "MOVE KEYFRAME:"
    VISION_KEYFRAME = "VISION KEYFRAME:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list(keyframe=False):
    if keyframe:
        return [
            CotTag.TASK.value,
            CotTag.PLAN.value,
            CotTag.VISIBLE_OBJECTS.value,
            CotTag.SUBTASK_REASONING.value,
            CotTag.SUBTASK.value,
            CotTag.MOVE_REASONING.value,
            CotTag.MOVE.value,
            CotTag.GRIPPER_POSITION.value,
            CotTag.SUBTASK_KEYFRAME.value,
            CotTag.MOVE_KEYFRAME.value,
            CotTag.VISION_KEYFRAME.value,
            CotTag.ACTION.value,
        ]
    else:
        return [
            CotTag.TASK.value,
            CotTag.PLAN.value,
            CotTag.VISIBLE_OBJECTS.value,
            CotTag.SUBTASK_REASONING.value,
            CotTag.SUBTASK.value,
            CotTag.MOVE_REASONING.value,
            CotTag.MOVE.value,
            CotTag.GRIPPER_POSITION.value,
            CotTag.ACTION.value,
        ]


def get_cot_database_keys(keyframe=False):
    if keyframe:
        return {
            CotTag.TASK.value: "task",
            CotTag.PLAN.value: "plan",
            CotTag.VISIBLE_OBJECTS.value: "bboxes",
            CotTag.SUBTASK_REASONING.value: "subtask_reason",
            CotTag.SUBTASK.value: "subtask",
            CotTag.MOVE_REASONING.value: "move_reason",
            CotTag.MOVE.value: "move",
            CotTag.GRIPPER_POSITION.value: "gripper",
            CotTag.SUBTASK_KEYFRAME.value: "subtask_keyframe",
            CotTag.MOVE_KEYFRAME.value: "move_keyframe",
            CotTag.VISION_KEYFRAME.value: "vision_keyframe",
            CotTag.ACTION.value: "action",
        }
    else:
        return {
            CotTag.TASK.value: "task",
            CotTag.PLAN.value: "plan",
            CotTag.VISIBLE_OBJECTS.value: "bboxes",
            CotTag.SUBTASK_REASONING.value: "subtask_reason",
            CotTag.SUBTASK.value: "subtask",
            CotTag.MOVE_REASONING.value: "move_reason",
            CotTag.MOVE.value: "move",
            CotTag.GRIPPER_POSITION.value: "gripper",
            CotTag.ACTION.value: "action",
        }
