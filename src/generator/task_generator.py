import random, json
from datetime import datetime
from pathlib import Path

GENERATED_DIR = Path("data/generated_tasks")
REGISTRY_PATH = Path("data/task_registry.json")

def generate_from_prompt(prompt: str, axis: str, level: str):
    """Generate a simple banking task from a free-text prompt for C1.1 or C1.2."""
    random.seed()  # ensure variability but log seed if needed
    
    # simple domain rules for consistency
    senders = ["Alice", "Bob", "Charlie"]
    receivers = ["Diana", "Eve", "Frank"]
    currencies = ["USD", "EUR"]
    amounts = [50, 100, 250, 1000]
    
    sender = random.choice(senders)
    receiver = random.choice(receivers)
    amount = random.choice(amounts)
    currency = random.choice(currencies)

    if level == "C1.1":
        steps = [
            {"instruction": "Open the banking app."},
            {"instruction": f"Transfer {amount} {currency} from {sender} to {receiver}."}
        ]
    elif level == "C1.2":
        steps = [
            {"instruction": "Check your current account balance."},
            {"instruction": f"Use that balance to transfer {amount} {currency} to {receiver}."}
        ]
    else:
        raise ValueError(f"Unsupported level {level} for Axis {axis}")
    
    task = {
        "task_id": f"task_{int(datetime.now().timestamp())}_{random.randint(100,999)}",
        "prompt": prompt,
        "axis": axis,
        "level": level,
        "steps": steps,
        "variables": {
            "sender": sender, "receiver": receiver,
            "amount": amount, "currency": currency
        },
        "created_at": datetime.now().isoformat()
    }
    
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATED_DIR / f"{task['task_id']}.json", "w") as f:
        json.dump(task, f, indent=2)
    
    # simplified registry logging
    if REGISTRY_PATH.exists():
        registry = json.load(open(REGISTRY_PATH))
    else:
        registry = []
    registry.append({"task_id": task["task_id"], "axis": axis, "level": level})
    json.dump(registry, open(REGISTRY_PATH, "w"), indent=2)
    
    return task
