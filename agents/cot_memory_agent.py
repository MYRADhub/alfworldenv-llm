import dspy
from typing import Literal

class CoTMemorySignature(dspy.Signature):
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

    observation: str = dspy.InputField(desc="Current environment observation.")
    seen_descriptions: str = dspy.InputField(desc="Descriptions of seen objects.")
    admissible_commands: list[str] = dspy.InputField(desc="Available actions.")
    memory: str = dspy.InputField(desc="Chronological history of prior observations and actions.")

    reasoning: str = dspy.OutputField(desc="Step-by-step logical reasoning.")
    action: str = dspy.OutputField(desc="Chosen next action.")
    prediction: Literal['professor', 'assassin', 'student', 'billionaire'] = dspy.OutputField(
        desc="Best guess for the profession.", default="unknown"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in prediction (0-10).", default=0.0
    )
    stop: bool = dspy.OutputField(
        desc="Whether to stop exploration.", default=False
    )

class CoTMemoryAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.ChainOfThought(CoTMemorySignature)
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
            print(f"⚠️ DSPy CoTMemoryAgent error: {e}")
            return "look around", "unknown", 0.0, False

        # Update memory
        self.memory_buffer.append(f"OBSERVED: {observation}")
        self.memory_buffer.append(f"ACTION: {result.action}")
        self.trim_buffer(self.memory_buffer, max_length=20)

        print("\n🧠 [CoT+Memory] Reasoning:\n", result.reasoning)
        print("🤖 [CoT+Memory] Chose action:", result.action)
        print("🔍 [CoT+Memory] Prediction:", result.prediction, f"({result.confidence:.2f} confidence)")
        print("🛑 [CoT+Memory] Wants to stop:", result.stop)

        return result.action, result.prediction, result.confidence, result.stop
    
    def trim_buffer(self, buffer, max_length=40):
        while len(buffer) > max_length:
            buffer.pop(0)
