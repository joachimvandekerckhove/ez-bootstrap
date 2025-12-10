#!/bin/bash

# EZ-CDM Simulation Runner
# Runs both single condition and design matrix simulations

SEED=1
N_REPETITIONS=10000
SAMPLE_SIZE=100
N_BOOTSTRAP=1000

cd "$(dirname "$0")"

echo "=========================================="
echo "EZ-CDM Simulation Study"
echo "=========================================="
echo "Running simulations with:"
echo "  Repetitions: $N_REPETITIONS"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Bootstrap samples: $N_BOOTSTRAP"
echo "  Seed: $SEED"
echo ""

# Run single condition simulation
echo "Running single condition simulation..."
./ezcdm_single --simulation $N_REPETITIONS $SAMPLE_SIZE $N_BOOTSTRAP --seed $SEED | tee ../output-ezcdm-single.txt

# Append design matrix simulation
echo "" >> ../output-ezcdm.txt
echo "==========================================" >> ../output-ezcdm.txt
echo "Design Matrix Simulation" >> ../output-ezcdm.txt
echo "==========================================" >> ../output-ezcdm.txt
echo "" >> ../output-ezcdm.txt

echo "Running design matrix simulation..."
./ezcdm_design_matrix --simulation $N_REPETITIONS $SAMPLE_SIZE $N_BOOTSTRAP --seed $SEED | tee ../output-ezcdm-design_matrix.txt

echo ""
echo "Simulations complete. Results saved to ../output-ezcdm-*.txt"
