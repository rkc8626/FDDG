#!/bin/bash

# Set variables
PYTHONPATH=/home/chenz1/toorange/FDDG/fddg
DATA_DIR=/home/chenz1/toorange/Data
OUT_DIR=/home/chenz1/toorange/FDDG/results
# LOG_DIR=/home/chenz1/toorange/FDDG/fddg/logs
SCRIPT=fddg.domainbed.scripts.train

# Create output directories
mkdir -p "$OUT_DIR"
# mkdir -p "$LOG_DIR"

# Datasets and Algorithms
# datasets=(CCMNIST1 FairFace NYPD YFCC)
datasets=(FairFace)
# algorithms=(ERM Fish GroupDRO IGA IRM Mixup SagNet)
algorithms=(IGA IRM Mixup SagNet)

# Prepare total tasks
total_tasks=$(( ${#datasets[@]} * ${#algorithms[@]} ))
completed_tasks=0
start_time=$(date +%s)

# Loop over all combinations
for dataset in "${datasets[@]}"; do
  for algo in "${algorithms[@]}"; do
    ((completed_tasks++))

    echo "=============================="
    echo "[$completed_tasks / $total_tasks] Running: $algo on $dataset"

    # Show ETA if at least one task is done
    if [ $completed_tasks -gt 1 ]; then
      current_time=$(date +%s)
      elapsed=$((current_time - start_time))
      avg_time=$((elapsed / (completed_tasks - 1)))
      remaining_tasks=$((total_tasks - completed_tasks + 1))
      eta_seconds=$((avg_time * remaining_tasks))
      eta_minutes=$((eta_seconds / 60))
      eta_formatted=$(date -d "+$eta_seconds seconds" +"%Y-%m-%d %H:%M:%S")
      echo "⏳ Estimated remaining time: ~$eta_minutes min (ETA: $eta_formatted)"
    fi

    # Build paths
    result_path="$OUT_DIR/results_${algo}_${dataset}"
    # log_path="$LOG_DIR/debug_${algo}_${dataset}.txt"

    # Run training
    PYTHONPATH=$PYTHONPATH uv run python -m $SCRIPT \
      --data_dir=$DATA_DIR \
      --dataset=$dataset \
      --algorithm=$algo \
      --test_env=0 \
      --output_dir=$result_path \
      # --save_predictions_every_checkpoint \
      # > "$log_path" 2>&1

    echo "✅ Finished: $algo on $dataset"
    echo
  done
done

echo "🎉 All $total_tasks experiments completed."
