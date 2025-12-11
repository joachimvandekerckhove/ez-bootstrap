#!/bin/bash

# Run all simulation studies for both EZ-DDM and EZ-CDM

cd "$(dirname "$0")"

echo "=========================================="
echo "Running All Simulation Studies"
echo "=========================================="
echo ""

# Make sure executables are built
echo "Building executables..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Build failed!"
    exit 1
fi

echo "Build complete."
echo ""

# Run EZ-DDM simulations
echo "=========================================="
echo "Starting EZ-DDM simulations..."
echo "=========================================="
cd ezddm
chmod +x run_simulations.sh
./run_simulations.sh
cd ..

echo ""
echo "=========================================="
echo "Starting EZ-CDM simulations..."
echo "=========================================="
cd ezcdm
chmod +x run_simulations.sh
./run_simulations.sh
cd ..

echo ""
echo "=========================================="
echo "All simulations complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - output-ezddm.txt"
echo "  - output-ezcdm.txt"
echo ""
