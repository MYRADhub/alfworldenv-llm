import dspy
from typing import Literal

class ExploreMemoryWithMap(dspy.Signature):
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

    Use history of actions, memory of previous steps, and a local map of opened/closed containers to plan exploration.
    """

    observation: str = dspy.InputField()
    seen_descriptions: str = dspy.InputField()
    admissible_commands: list[str] = dspy.InputField()
    memory: str = dspy.InputField()
    local_map: str = dspy.InputField()

    action: str = dspy.OutputField()
    prediction: Literal['professor', 'assassin', 'student', 'billionaire'] = dspy.OutputField(default="unknown")
    confidence: float = dspy.OutputField(default=0.0)
    stop: bool = dspy.OutputField(default=False)

class MemoryMapAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.Predict(ExploreMemoryWithMap)
        self.memory_buffer = []
        self.map_buffer = []

    def forward(self, observation, seen_descriptions, admissible_commands):
        memory_text = "\n".join(self.memory_buffer)
        map_text = "\n".join(self.map_buffer)

        try:
            result = self.policy(
                observation=observation,
                seen_descriptions="\n".join(seen_descriptions),
                admissible_commands=admissible_commands,
                memory=memory_text,
                local_map=map_text
            )
        except Exception as e:
            print(f"⚠️ DSPy error: {e}")
            return "look around", "unknown", 0.0, False

        # Update memory
        self.memory_buffer.append(f"OBSERVED: {observation}")
        self.memory_buffer.append(f"ACTION: {result.action}")
        self.trim_buffer(self.memory_buffer, max_length=40)

        print("\n🧠 [Memory+Map] Chose action:", result.action)
        print("🔍 [Memory+Map] Prediction:", result.prediction, f"({result.confidence:.2f} confidence)")
        print("🛑 [Memory+Map] Wants to stop:", result.stop)

        return result.action, result.prediction, result.confidence, result.stop

    def update_map(self, observation, action):
        self.map_buffer.append(f"ACTION: {action}")
        self.map_buffer.append(f"OBSERVED: {observation}")
        self.trim_buffer(self.map_buffer, max_length=40)

    def trim_buffer(self, buffer, max_length=40):
        while len(buffer) > max_length:
            buffer.pop(0)
