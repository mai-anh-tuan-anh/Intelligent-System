# Read a trained AGENT02 tabular Q-learning policy from model.json.
from pathlib import Path
import json


def choose_action(demand: int, competitor_previous_action: int, model_file=None) -> int:
    # Return 0 (no discount) or 1 (discount) for a given state.
    if demand not in (0, 1) or competitor_previous_action not in (0, 1):
        raise ValueError("demand and competitor_previous_action must be 0 or 1")
    path = Path(model_file) if model_file else Path(__file__).with_name("model.json")
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    values = model["q_table"][demand][competitor_previous_action]
    return int(max(range(len(values)), key=lambda a: values[a]))


if __name__ == "__main__":
    for demand in (0, 1):
        for previous_action in (0, 1):
            print((demand, previous_action), "->", choose_action(demand, previous_action))
