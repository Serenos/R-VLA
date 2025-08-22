import os
import sys
import json
import uuid

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "default")
CACHE_DIR = f"sim_cogact/sim_inference_cfgs/gpu_{gpu_id}"

os.makedirs(CACHE_DIR, exist_ok=True)
cache_file = os.path.join(CACHE_DIR, f"args_{uuid.uuid4().hex}.json")
with open(cache_file, "w") as f:
    json.dump(sys.argv[1:], f, indent=2)
print(f"Args已缓存到：{cache_file}")
