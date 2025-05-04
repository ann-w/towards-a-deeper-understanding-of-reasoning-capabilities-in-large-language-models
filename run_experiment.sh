#!/bin/bash

# Default experiment name
experiment_name="orca2-13b"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) experiment_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src    

# Create the experiments_logs directory if it doesn't exist
mkdir -p experiments_logs

# Get the current date in YYYYMMDD format
current_date=$(date +"%Y%m%d")

# Define the log file path
log_file="experiments_logs/${current_date}_${experiment_name}.log"

# Run the Python script and redirect logs
nohup python src/scripts/main.py > "$log_file" 2>&1 

# Get the PID of the last background process
pid=$!

# Disown the process to ensure it keeps running after the terminal is closed
disown $pid