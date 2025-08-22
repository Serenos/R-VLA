import json
from tqdm import tqdm
import numpy as np

bridge_reasoning_path = "/data/lixiang10/embodiedCoT/embodied_features_bridge.json"

with open(bridge_reasoning_path, "r") as f:
    reasoning_data = json.load(f)

print("len bridge_reasoning keys:", len(reasoning_data.keys()))


def gaussian_smooth(keyframe_list, window_size=3, sigma=1.0, threshold=0.1):
    # 生成高斯核
    kernel = np.exp(-0.5 * (np.arange(window_size) - window_size // 2) ** 2 / sigma**2)
    kernel /= kernel.sum()
    # 卷积平滑
    smoothed = np.convolve(keyframe_list, kernel, mode="same")
    # 阈值化，非零即关键帧
    return [1 if v > threshold else 0 for v in smoothed]


if __name__ == "__main__":
    use_gaussian_smooth = True
    window_size = 2

    for filename in tqdm(reasoning_data.keys()):
        for demo_id in reasoning_data[filename].keys():
            demo = reasoning_data[filename][demo_id]
            subtask_keyframe = []
            move_keyframe = []
            vision_keyframe = []
            cur_subtask, cur_move = "", ""
            cur_vision = set()
            if "reasoning" not in demo.keys():
                continue
            for key, value in demo["reasoning"].items():
                move = value["move"]
                subtask = value["subtask"]
                if "bboxes" in demo["features"] and int(key) < len(demo["features"]["bboxes"]):
                    vision = demo["features"]["bboxes"][int(key)]
                else:
                    vision = []
                # print(move)
                if move != cur_move:
                    cur_move = move
                    move_keyframe.append(1)
                else:
                    move_keyframe.append(0)

                # print(subtask)
                if subtask != cur_subtask:
                    cur_subtask = subtask
                    subtask_keyframe.append(1)
                else:
                    subtask_keyframe.append(0)

                visble_objs_set = set()
                # print(vision)
                if len(vision) == 0:
                    vision_keyframe.append(0)
                    continue
                for obj in vision:
                    for info in obj:
                        if type(info) is str:
                            visble_objs_set.add(info)
                # print(visble_objs_set)
                if visble_objs_set != cur_vision:
                    cur_vision = visble_objs_set
                    vision_keyframe.append(1)
                else:
                    vision_keyframe.append(0)
            if use_gaussian_smooth:
                if len(subtask_keyframe) != 0:
                    subtask_keyframe = gaussian_smooth(subtask_keyframe, window_size=window_size, sigma=1.0, threshold=0.1)
                if len(move_keyframe) != 0:
                    move_keyframe = gaussian_smooth(move_keyframe, window_size=window_size, sigma=1.0, threshold=0.1)
                # vision_keyframe = gaussian_smooth(vision_keyframe, window_size=3, sigma=1.0, threshold=0.1)

            demo["features"]["subtask_keyframe"] = subtask_keyframe
            demo["features"]["move_keyframe"] = move_keyframe
            demo["features"]["vision_keyframe"] = vision_keyframe

    with open(f"/data/lixiang10/embodiedCoT/embodied_features_bridge_keyframe_gauss_w{window_size}.json", "w") as f:
        json.dump(reasoning_data, f, indent=4)
    print(
        f"Keyframe annotation completed and saved to /data/lixiang10/embodiedCoT/embodied_features_bridge_keyframe_gauss_w{window_size}.json"
    )

    print(demo["features"]["move_primitive"])
    print(demo["features"]["move_keyframe"])
    print(demo["features"]["subtask_keyframe"])
