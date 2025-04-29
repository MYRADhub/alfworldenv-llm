import json
import re
import random

import alfworld.agents.modules.generic as generic
from alfworld.agents.environment import get_environment
import agents as agents

import dspy
import os
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
        raise RuntimeError("No task found!")

    setattr(env, file_list_attr, kept)
    env.num_games = len(kept)
    return env

# Load extra attributes
with open("extra_attributes.json") as f:
    extra_attributes = json.load(f)

# Set up environment
config = generic.load_config()
env_type = config['env']['type']
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)
restrict_environment(env, number=1)

# Start episode
obs, info = env.reset()
lines = obs[0].split('\n')
if lines and lines[-1].strip().lower().startswith("your task is to"):
    obs = ['\n'.join(lines[:-1])]

print("🧭 Initial Observation:\n", obs[0])
print("\n🧩 Admissible commands:\n", info["admissible_commands"][0])

agent = agents.ReasoningAgent()
seen_descriptions = {}
step_counter = 0
conf_threshold = 7.5
profession = None
confidence = 0.0
action_history = []

while True:
    cmds = info["admissible_commands"][0]
    
    action, profession, confidence, stop = agent(
        observation=obs[0],
        seen_descriptions=list(seen_descriptions.values()),
        admissible_commands=cmds
    )

    # Avoid repetition
    if action in action_history[-3:]:
        print("🔁 Detected repetition. Choosing random fallback action.")
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

    if confidence >= conf_threshold or stop:
        if stop:
            print("🛑 Agent decided to stop exploring.")
        else:
            print("🔍 Agent reached high confidence threshold.")
        print(f"\n✅ Agent stopped with high confidence: {profession} ({confidence:.1f}) after {step_counter} steps.")
        break

    if dones[0]:
        print("\n🏁 Episode finished.")
        break

# Final report
print(f"\n🔍 Final prediction: {profession} ({confidence:.1f} confidence)")
print(f"🧮 Total steps taken: {step_counter}")
