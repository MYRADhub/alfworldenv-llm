#!/bin/bash

# List of agent types
AGENTS=("naive" "memory" "cot_memory" "cot" "naive_map" "memory_map" "cot_map" "cot_memory_map")

# Create a logs directory
mkdir -p logs

# Loop through each agent
for AGENT in "${AGENTS[@]}"; do
    echo "🚀 Starting evaluation for agent: $AGENT"
    python main.py base_config.yaml --agent $AGENT --floorplan_random > "logs/${AGENT}_log.txt" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Agent $AGENT completed successfully."
    else
        echo "❌ Agent $AGENT failed. Check logs/${AGENT}_log.txt for details."
    fi
done

echo "🏁 All agent evaluations finished."
