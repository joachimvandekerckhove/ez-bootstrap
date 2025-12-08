#!/bin/bash

SEED=100
N_REPETITIONS=10000
SAMPLE_SIZE=100
N_BOOTSTRAP=1000

cd src

make

# Run single condition simulation
./ezb_single --simulation $N_REPETITIONS $SAMPLE_SIZE $N_BOOTSTRAP --seed $SEED

# Run design matrix simulation
./ezb_design_matrix --simulation $N_REPETITIONS $SAMPLE_SIZE $N_BOOTSTRAP  --seed $SEED
