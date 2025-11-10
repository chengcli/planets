#! /bin/bash
# Step 1: Generate configuration file for White Sands simulation
python generate_config.py white-sands \
    --start-date 2025-10-01 --end-date 2025-10-02 \
    --nx1 150 --nx2 1600 --nx3 1200 \
    --output white-sands.yaml

# Step 2: Prepare initial condition data
python prepare_initial_condition.py white-sands --nX 4 --nY 4

# Step 3: Run simulations

# Step 4: Calculate diagnostics and KPIs
