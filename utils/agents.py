import dspy
from typing import Literal

class ExploreAndGuess(dspy.Signature):
    """
    You are an intelligent agent exploring a household in ALFWorld.

    Your job is to determine the likely profession of the room's occupant 
    (one of: professor, assassin, student, billionaire), **but only after gathering enough evidence**.

    Associate the profession with the objects you see. (think about what objects might each profession have)

    Your goal is to guess the profession of the occupant based on observed items.

    🎯 Confidence is from 0 to 10:
    - 10 = Absolute certainty.
    - 0 = No idea.

    🔚 If you're confident enough and can justify your guess based on seen objects, you may stop.
    """

    observation: str = dspy.InputField(desc="What the agent currently sees.")
    seen_descriptions: str = dspy.InputField(desc="Descriptions of previously seen objects.")
    admissible_commands: list[str] = dspy.InputField(desc="List of valid actions to choose from.")

    action: str = dspy.OutputField(desc="Most informative next action to take.")
    prediction: Literal['professor', 'assassin', 'student','billionaire'] = dspy.OutputField(
        desc="Your current best guess of the profession.", default="unknown"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the guess (0 = unsure, 10 = very sure).", default=0.0
    )
    stop: bool = dspy.OutputField(
        desc="Should the agent stop exploring and finalize the guess?", default=False
    )

class ReasoningAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.Predict(ExploreAndGuess)

    def forward(self, observation, seen_descriptions, admissible_commands):
        # print("🔍 [Agent] Seen descriptions:", seen_descriptions)
        # print("🔍 [Agent] Admissible commands:", admissible_commands)
        # print("🔍 [Agent] Current observation:", observation)

        try:
            result = self.policy(
                observation=observation,
                seen_descriptions="\n".join(seen_descriptions),
                admissible_commands=admissible_commands
            )
        except Exception as e:
            print("⚠️ DSPy Prediction failed:", e)
            return "look", "unknown", 0.0, False

        print("\n🤖 [Agent] Chose action:", result.action)
        print("🔍 [Agent] Prediction:", result.prediction, f"({result.confidence:.2f} confidence)")
        print("🛑 [Agent] Wants to stop:", result.stop)

        return result.action, result.prediction, result.confidence, result.stop
