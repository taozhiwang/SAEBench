#!/bin/bash

# Script to run sae_bench_eval.py with all combinations of sae_key (1-8) and gpu_id (0-7)
# The script runs each evaluation in parallel

echo "Starting SAE evaluations with all key-GPU combinations..."

for sae_key in {1..10}; do
  # Calculate GPU ID based on SAE key (1-indexed to 0-indexed)
  gpu_id=$((sae_key - 1))
  
  # Convert GPU ID to string
  gpu_id="${gpu_id}"
  
  # Create log file
  LOG_FILE="logs/sae_eval_key${sae_key}_gpu${gpu_id}.log"
  mkdir -p logs
  # Log the start of each evaluation
  echo "Running evaluation with SAE key: $sae_key on GPU: $gpu_id"
  
  # Run the Python script with the current parameters
  tmux new-window -t "sae" "bash -c 'export CUDA_VISIBLE_DEVICES=${gpu_id}; python sae_bench_eval.py --sae_key ${sae_key} --gpu_id ${gpu_id} | tee ${LOG_FILE}'"
  
  # Optional: add a small delay to prevent all processes from starting at exactly the same time
  sleep 1
done

# Wait for all background processes to complete
wait

echo "All evaluations completed!" 