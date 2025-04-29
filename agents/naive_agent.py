import dspy
from typing import Literal

class ExploreAndGuess(dspy.Signature):
    """
    You are an intelligent agent exploring a house to guess the profession of the resident.

    Your job is to determine the likely profession of the room's occupant 
    (one of: professor, assassin, student, billionaire), **but only after gathering enough evidence**.

    Associate the profession with the objects you see. (think about what objects might each profession have)

    Your goal is to guess the profession of the occupant based on observed items.

    Confidence is from 0 to 10:
    - 10 = Absolute certainty.
    - 0 = No idea.

    If you're confident enough and can justify your guess based on seen objects, you may stop.
    """

    observation: str = dspy.InputField(desc="Current visible environment description.")
    seen_descriptions: str = dspy.InputField(desc="Descriptions of discovered objects so far.")
    admissible_commands: list[str] = dspy.InputField(desc="Available actions to choose from.")

    action: str = dspy.OutputField(desc="Most useful next action to explore.")
    prediction: Literal['professor', 'assassin', 'student', 'billionaire'] = dspy.OutputField(
        desc="Best guess of profession so far.", default="unknown"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in prediction (0 to 10).", default=0.0
    )
    stop: bool = dspy.OutputField(
        desc="Should exploration stop (True if confident)?", default=False
    )

class NaiveAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.Predict(ExploreAndGuess)

    def forward(self, observation, seen_descriptions, admissible_commands):
        try:
            result = self.policy(
                observation=observation,
                seen_descriptions="\n".join(seen_descriptions),
                admissible_commands=admissible_commands
            )
        except Exception as e:
            print(f"⚠️ DSPy error: {e}")
            return "look around", "unknown", 0.0, False

        print("\n🤖 [Agent] Chose action:", result.action)
        print("🔍 [Agent] Prediction:", result.prediction, f"({result.confidence:.2f} confidence)")
        print("🛑 [Agent] Wants to stop:", result.stop)

        return result.action, result.prediction, result.confidence, result.stop
