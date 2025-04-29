# Alfworld extra imports
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment import get_environment

# Configure DSPy with OpenAI's GPT model
import dspy
lm = dspy.LM(model='gpt-4o-mini', api_key='key')
dspy.configure(lm=lm)

# Define a DSPy module to choose actions
class ALFWorldAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize your dspy module here, e.g., dspy.ReAct
        self.react = dspy.ReAct("observation, admissible_commands -> action")


    def forward(self, observation, admissible_commands):
        # TODO: Complete the forward method with your exploration method
        result = self.react(observation=observation,
                            admissible_commands=admissible_commands)
        return result.action

# Load environment configuration
config = generic.load_config()
env_type = config['env']['type']

# Set up environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# Reset environment to initial state
obs, info = env.reset()

# Initialize DSPy ALFWorld agent
agent = ALFWorldAgent()

# Action-selection loop
while True:
    admissible_commands = info['admissible_commands'][0]

    # Obtain next action using DSPy
    action = agent(observation=obs[0], admissible_commands=admissible_commands)
    print(f"Action chosen by DSPy agent: {action}")

    # Execute action in environment
    obs, scores, dones, info = env.step([action])
    print(f"Observation: {obs[0]}")

    # Check for episode completion
    if dones[0]:
        print("Episode completed.")
        break
