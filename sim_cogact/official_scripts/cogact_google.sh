#!/bin/bash

scripts=(
#  "sim_cogact/official_scripts/cogact_drawer_variant_agg.sh"
#  "sim_cogact/official_scripts/cogact_drawer_visual_matching.sh"
#  "sim_cogact/official_scripts/cogact_move_near_variant_agg.sh"
#  "sim_cogact/official_scripts/cogact_move_near_visual_matching.sh"
#  "sim_cogact/official_scripts/cogact_pick_coke_can_variant_agg.sh"
#  "sim_cogact/official_scripts/cogact_pick_coke_can_visual_matching.sh"
  "sim_cogact/official_scripts/cogact_put_in_drawer_variant_agg.sh"
  "sim_cogact/official_scripts/cogact_put_in_drawer_visual_matching.sh"
)

mkdir -p logs

for s in "${scripts[@]}"; do
  echo "Starting $s..."
  bash "$s" > "logs/cogact_google.log" 2>&1 &
done

wait

echo "All processes have completed."
