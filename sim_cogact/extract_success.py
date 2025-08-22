import os
import re

def extract_success_rates(log_path: str):
    pattern = re.compile(r"Average success:?\s+([0-9]*\.?[0-9]+)")
    rates = []
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rates.append(float(m.group(1)))
    return rates

if __name__ == "__main__":
    folder = "/data/ckpt/cogact/pretrained/CogACT-Base/eval/CogACT-Base.pt"

    files = [
        "google_pick_coke_can_visual_matching.txt",
        "google_move_near_visual_matching.txt",
        "google_drawer_visual_matching.txt",
        "google_put_in_drawer_visual_matching.txt",
        "google_pick_coke_can_variant_agg.txt",
        "google_move_near_variant_agg.txt",
        "google_drawer_variant_agg.txt",
        "google_put_in_drawer_variant_agg.txt",
    ]

    for fname in files:
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            print(f"{fname} 不存在，跳过")
            continue

        rates = extract_success_rates(path)
        if not rates:
            print(f"{fname}：未找到成功率数据")
            continue

        avg_rate = sum(rates) / len(rates)
        rates_str = ", ".join(f"{r:.6f}" for r in rates)
        print(f"\n{fname}")
        print(f"  成功率列表: [{rates_str}]")
        print(f"  最终成功率（平均）: {avg_rate:.6f}")

