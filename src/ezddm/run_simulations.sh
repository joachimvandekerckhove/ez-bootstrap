#!/bin/bash

# EZ-DDM Simulation Runner
# Runs both single condition and design matrix simulations

SEED=1
N_REPETITIONS=10000
SAMPLE_SIZE=100
N_BOOTSTRAP=1000

cd "$(dirname "$0")"

echo "=========================================="
echo "EZ-DDM Simulation Study"
echo "=========================================="
echo "Running simulations with:"
echo "  Repetitions: $N_REPETITIONS"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Bootstrap samples: $N_BOOTSTRAP"
echo "  Seed: $SEED"
echo ""

# Run single condition simulation
echo "Running single condition simulation..."
./ezb_single --simulation $N_REPETITIONS $SAMPLE_SIZE $N_BOOTSTRAP --seed $SEED | tee ../output-ezddm-single.txt

# Append design matrix simulation
echo "" >> ../output-ezddm.txt
echo "==========================================" >> ../output-ezddm.txt
echo "Design Matrix Simulation" >> ../output-ezddm.txt
echo "==========================================" >> ../output-ezddm.txt
echo "" >> ../output-ezddm.txt

echo "Running design matrix simulation..."
./ezb_design_matrix --simulation $N_REPETITIONS $SAMPLE_SIZE $N_BOOTSTRAP --seed $SEED | tee ../output-ezddm-design_matrix.txt

echo ""
echo "Simulations complete. Results saved to ../output-ezddm-*.txt"
