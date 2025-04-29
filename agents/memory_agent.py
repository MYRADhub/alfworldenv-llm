import dspy
from typing import Literal

class ExploreWithMemory(dspy.Signature):
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

    observation: str = dspy.InputField(desc="Current observation in the environment.")
    seen_descriptions: str = dspy.InputField(desc="Descriptions of all objects seen so far.")
    admissible_commands: list[str] = dspy.InputField(desc="Available actions you can take now.")
    memory: str = dspy.InputField(desc="Chronological memory of previous actions and observations.")

    action: str = dspy.OutputField(desc="Best next action to maximize information.")
    prediction: Literal['professor', 'assassin', 'student', 'billionaire'] = dspy.OutputField(
        desc="Best guess for the profession.", default="unknown"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score (0-10).", default=0.0
    )
    stop: bool = dspy.OutputField(
        desc="Should the exploration stop now?", default=False
    )

class MemoryAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.Predict(ExploreWithMemory)
        self.memory_buffer = []

    def forward(self, observation, seen_descriptions, admissible_commands):
        memory_text = "\n".join(self.memory_buffer)

        try:
            result = self.policy(
                observation=observation,
                seen_descriptions="\n".join(seen_descriptions),
                admissible_commands=admissible_commands,
                memory=memory_text
            )
        except Exception as e:
            print(f"⚠️ DSPy error: {e}")
            return "look around", "unknown", 0.0, False

        # Update memory
        self.memory_buffer.append(f"OBSERVED: {observation}")
        self.memory_buffer.append(f"ACTION: {result.action}")
        self.trim_buffer(self.memory_buffer, max_length=40)

        print("\n🧠 [MemoryAgent] Chose action:", result.action)
        print("🔍 [MemoryAgent] Prediction:", result.prediction, f"({result.confidence:.2f} confidence)")
        print("🛑 [MemoryAgent] Wants to stop:", result.stop)

        return result.action, result.prediction, result.confidence, result.stop

    def trim_buffer(self, buffer, max_length=40):
        while len(buffer) > max_length:
            buffer.pop(0)