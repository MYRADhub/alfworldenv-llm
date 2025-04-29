import dspy
from typing import Literal

class CoTSignature(dspy.Signature):
    """
    You are an intelligent agent exploring a house to guess the profession of the resident.

    Your job is to determine the likely profession of the room's occupant 
    (one of: professor, assassin, student, billionaire), **but only after gathering enough evidence**.

    Associate the profession with the objects you see. (think about what objects might each profession have)

    Your goal is to guess the profession of the occupant based on observed items.

    At each step:
    - Analyze your current observation.
    - Consider all previously seen object descriptions.
    - Think step-by-step to decide on the most informative next action.
    - If confident enough, predict the profession and explain your reasoning.

    Confidence is from 0 to 10:
    - 10 = Absolute certainty.
    - 0 = No idea.

    If you're confident enough and can justify your guess based on seen objects, you may stop.
    """

    observation: str = dspy.InputField(desc="Current observation in the environment.")
    seen_descriptions: str = dspy.InputField(desc="Descriptions of previously seen objects.")
    admissible_commands: list[str] = dspy.InputField(desc="List of valid actions to choose from.")

    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process.")
    action: str = dspy.OutputField(desc="Chosen next action.")
    prediction: Literal['professor', 'assassin', 'student', 'billionaire'] = dspy.OutputField(
        desc="Predicted profession based on gathered evidence.", default="unknown"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the prediction (0 = unsure, 10 = very sure).", default=0.0
    )
    stop: bool = dspy.OutputField(
        desc="Whether to stop exploring and finalize the prediction.", default=False
    )

class CoTAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.ChainOfThought(CoTSignature)

    def forward(self, observation, seen_descriptions, admissible_commands):
        try:
            result = self.policy(
                observation=observation,
                seen_descriptions="\n".join(seen_descriptions),
                admissible_commands=admissible_commands
            )
        except Exception as e:
            print("⚠️ DSPy ChainOfThought Prediction failed:", e)
            return "look around", "unknown", 0.0, False

        print("\n🧠 [CoT] Reasoning:\n", result.reasoning)
        print("🤖 [CoT] Chose action:", result.action)
        print("🔍 [CoT] Prediction:", result.prediction, f"({result.confidence:.2f} confidence)")
        print("🛑 [CoT] Wants to stop:", result.stop)

        return result.action, result.prediction, result.confidence, result.stop
