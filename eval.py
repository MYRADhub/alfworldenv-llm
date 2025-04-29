import json
import re
import random
import os

from alfworld.agents.environment import get_environment

# Import all agents
from agents.naive_agent import NaiveAgent
from agents.memory_agent import MemoryAgent
from agents.cot_agent import CoTAgent
from agents.cot_memory_agent import CoTMemoryAgent
from agents.naive_map_agent import NaiveMapAgent
from agents.memory_map_agent import MemoryMapAgent
from agents.cot_map_agent import CoTMapAgent
from agents.cot_memory_map_agent import CoTMemoryMapAgent

import dspy
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
lm = dspy.LM(model='gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

def restrict_environment(env, number: int = 1, mode: str = 'scene'):
    if hasattr(env, "json_file_list"):
        file_list_attr = "json_file_list"
    elif hasattr(env, "task_file_list"):
        file_list_attr = "task_file_list"
    elif hasattr(env, "game_file_list"):
        file_list_attr = "game_file_list"
    elif hasattr(env, "gamefiles"):
        file_list_attr = "gamefiles"
    else:
        raise RuntimeError("Unknown env file list attr")

    file_list = getattr(env, file_list_attr)

    def is_match(path: str) -> bool:
        if re.search(fr'FloorPlan{number}(?:[^0-9]|$)', path):
            return True
        m = re.search(r'-([0-9]{1,3})/', path)
        return bool(m and int(m.group(1)) == number)

    kept = [p for p in file_list if is_match(p)]

    if not kept:
        print(f"Warning: No tasks found for floorplan {number} in {file_list_attr}.")
        print(f"Available tasks: {file_list}")
        print(f"Regex used: {re.escape(fr'FloorPlan{number}(?:[^0-9]|$)')}")
        print(f"Regex used: {re.escape(r'-([0-9]{1,3})/')}")
        print(f"Regex used: {re.escape(r'FloorPlan([0-9]{1,3})')}")
        raise RuntimeError("No task found!")
    
    print(f"Restricting environment to floorplan {number} ({len(kept)} tasks found).")

    setattr(env, file_list_attr, kept)
    env.num_games = len(kept)
    return env

def run_episode(extra_attr_path="eval_attributes/extra_attributes.json", config=None, floorplan_number=1, conf_threshold=7.5, agent_type="naive", randomize_floorplan=False):
    assert config is not None, "You must pass a config dictionary to run_episode!"

    # Load object attributes
    with open(extra_attr_path) as f:
        extra_attributes = json.load(f)

    # Initialize environment
    env_type = config['env']['type']
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    if not randomize_floorplan:
        restrict_environment(env, number=floorplan_number)

    obs, info = env.reset()
    lines = obs[0].split('\n')
    if lines and lines[-1].strip().lower().startswith("your task is to"):
        obs = ['\n'.join(lines[:-1])]

    # Select agent
    agent_lookup = {
        "naive": NaiveAgent,
        "memory": MemoryAgent,
        "cot": CoTAgent,
        "cot_memory": CoTMemoryAgent,
        "naive_map": NaiveMapAgent,
        "memory_map": MemoryMapAgent,
        "cot_map": CoTMapAgent,
        "cot_memory_map": CoTMemoryMapAgent
    }

    if agent_type not in agent_lookup:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    agent = agent_lookup[agent_type]()
    seen_descriptions = {}
    action_history = []
    step_counter = 0
    profession = None
    confidence = 0.0

    while True:
        cmds = info["admissible_commands"][0]
        action, profession, confidence, stop = agent(
            observation=obs[0],
            seen_descriptions=list(seen_descriptions.values()),
            admissible_commands=cmds
        )

        # Avoid repetition
        if action in action_history[-3:]:
            cmds = [cmd for cmd in cmds if cmd != action]
            if cmds:
                action = random.choice(cmds)
        action_history.append(action)

        obs, scores, dones, info = env.step([action])
        step_counter += 1
        obs_text = obs[0].lower()

        for obj_key, obj_info in extra_attributes.items():
            if obj_key.lower() in obs_text and obj_key not in seen_descriptions:
                seen_descriptions[obj_key] = obj_info["description"]
                print(f"📦 Found new object: {obj_key} — {obj_info['description']}")

        if hasattr(agent, "update_map"):
            agent.update_map(obs[0], action)

        if confidence >= conf_threshold or stop:
            print(f"\n✅ Agent stopped after {step_counter} steps. Prediction: {profession} ({confidence:.1f})")
            break

        if dones[0]:
            print("\n🏁 Episode finished.")
            break

    env.close()
    if hasattr(env, "stop_unity"):
        env.stop_unity()

    return {
        "prediction": profession,
        "confidence": confidence,
        "steps": step_counter
    }


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config (e.g., base_config.yaml)")
    parser.add_argument("--attributes", type=str, default="eval_attributes/extra_attributes.json", help="Path to extra attributes JSON file.")
    parser.add_argument("--agent", type=str, default="naive", 
                        choices=["naive", "memory", "cot", "cot_memory", "naive_map", "memory_map", "cot_map", "cot_memory_map"],
                        help="Which agent to use.")
    parser.add_argument("--floorplan", type=int, default=1, help="Which floorplan number to use.")
    parser.add_argument("--conf_threshold", type=float, default=7.5, help="Confidence threshold to stop.")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = run_episode(
        extra_attr_path=args.attributes,
        config=config,
        floorplan_number=args.floorplan,
        conf_threshold=args.conf_threshold,
        agent_type=args.agent
    )

    print("\nFinal Result:", result)
