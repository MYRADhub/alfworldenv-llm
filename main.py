import glob
import os
import sys
import pandas as pd
import yaml
import random

from eval import run_episode

# Constants
ATTR_DIR = "./eval_attributes"
GROUND_TRUTH_LABELS = ["professor", "assassin", "student", "billionaire"]
TOTAL_RUNS = 20
AGENT_TYPES = ["naive", "memory", "cot", "cot_memory", "naive_map", "memory_map", "cot_map", "cot_memory_map"]

def extract_ground_truth(filename):
    for label in GROUND_TRUTH_LABELS:
        if label in filename.lower():
            return label
    return "unknown"

def load_config_from_cmd():
    assert len(sys.argv) > 1, "Please provide config YAML path, e.g.: python main.py base_config.yaml"
    config_path = sys.argv[1]
    with open(config_path) as f:
        return yaml.safe_load(f)

def batch_evaluate(config, agent_type="naive", randomize_floorplan=True):
    results = []
    attribute_files = sorted(glob.glob(os.path.join(ATTR_DIR, "*_attributes.json")))

    if not attribute_files:
        print("❌ No attribute files found matching '*_attributes.json'")
        return

    print(f"\n🧪 Running {TOTAL_RUNS} randomized episodes on agent '{agent_type}'...\n")

    for i in range(TOTAL_RUNS):
        file_path = random.choice(attribute_files)
        ground_truth = extract_ground_truth(file_path)
        floorplan_number = random.randint(1, 9) if randomize_floorplan else 1

        print(f"🎯 [Run {i+1}/{TOTAL_RUNS}] Agent: {agent_type} | Attr File: {os.path.basename(file_path)} | Floor: {floorplan_number} (GT: {ground_truth})")

        result = run_episode(
            extra_attr_path=file_path,
            config=config,
            floorplan_number=floorplan_number,   # This is ignored if randomizing
            conf_threshold=7.5,
            agent_type=agent_type,
            randomize_floorplan=randomize_floorplan
        )

        result["ground_truth"] = ground_truth
        result["file"] = os.path.basename(file_path)
        result["floorplan"] = floorplan_number
        result["agent_type"] = agent_type
        result["correct"] = (result["prediction"] == ground_truth)
        results.append(result)

    # Save
    df = pd.DataFrame(results)
    out_path = f"results/evaluation_results_{agent_type}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 Results saved to {out_path}")

    return df

def full_multiagent_benchmark(config, randomize_floorplan=True):
    all_results = []

    for agent_type in AGENT_TYPES:
        print(f"\n🚀 Starting benchmark for agent: {agent_type}")
        df = batch_evaluate(config, agent_type=agent_type, randomize_floorplan=randomize_floorplan)
        all_results.append(df)

    # Merge all results
    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv("full_benchmark_results.csv", index=False)

    print("\n📚 Full benchmark saved to 'full_benchmark_results.csv'.")

    # Summary
    print("\n🔍 Summary Report:")
    summary = full_df.groupby("agent_type").agg({
        "correct": ["sum", "count", lambda x: 100 * x.sum() / x.count()],
        "steps": "mean",
        "confidence": "mean"
    })
    summary.columns = ["# Correct", "# Total", "Accuracy (%)", "Avg Steps", "Avg Confidence"]
    print(summary)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to base_config.yaml")
    parser.add_argument("--agent", type=str, choices=AGENT_TYPES + ["all"], default="naive", help="Which agent to evaluate. Use 'all' for full benchmark.")
    parser.add_argument("--floorplan_random", action="store_true", help="Randomize floorplan between 1 and 30 each run.")
    args = parser.parse_args()

    config = load_config_from_cmd()

    if args.agent == "all":
        full_multiagent_benchmark(config, randomize_floorplan=args.floorplan_random)
    else:
        batch_evaluate(config, agent_type=args.agent, randomize_floorplan=args.floorplan_random)
