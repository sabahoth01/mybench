import json
from pathlib import Path

from src.generator.task_generator import generate_multiple_instances
from src.core.runner import run_trial_series

TEST_CATEGORY_FILE = Path("configs/test_category.json")

with open(TEST_CATEGORY_FILE, "r", encoding="utf-8") as f:
    TEST_CATEGORIES = json.load(f)

ORDERED_CATEGORIES = [
    (i, key, info)
    for i, (key, info) in enumerate(TEST_CATEGORIES.items(), start=1)
]

def interactive_mode():
    print("\nProcedural Memory Benchmark\n")

    for num, key, info in ORDERED_CATEGORIES:
        print(f"{num}. {info['name']} ({key}) – {info['description']}")

    try:
        selection = int(input("\nCategory number → ").strip())
    except ValueError:
        print("Invalid number.")
        return
    if not (1 <= selection <= len(ORDERED_CATEGORIES)):
        print("Invalid category.")
        return

    _, category_key, category_info = ORDERED_CATEGORIES[selection - 1]
    category_name = category_info["name"]

    task_prompt = input(
        f"\nEnter base task prompt for '{category_name}':\n→ "
    ).strip()

    try:
        n_instances = int(input("\nHow many task instances? → ").strip())
        n_trials = int(input("How many trials per instance? → ").strip())
    except ValueError:
        print("Invalid number.")
        return

    feedback_policy = input(
        "\nFeedback policy (minimal, //still testing not working yet):\n→ "
    ).strip()

    perturb_policy = input(
        "\nPerturbation policy (none, , //still testing not working yet):\n→ "
    ).strip()

    print(f"\nGenerating {n_instances} tasks…")
    tasks = generate_multiple_instances(
        prompt=task_prompt,
        category_key=category_key,
        category_name=category_name,
        n_instances=n_instances
    )

    print(f"\nGenerated {len(tasks)} tasks.")

    start = input("\nStart trials now? (y/n) → ").lower().strip()
    if start not in ("y", "yes"):
        print("Exiting.")
        return

    for idx, task in enumerate(tasks, start=1):
        print(f"\nTask {idx}/{len(tasks)} – {task['task_id']}")
        run_trial_series(
            task,
            n_trials,
            feedback_policy=feedback_policy,
            perturb_policy=perturb_policy
        )


def config_mode(path):
    print(f"Batch mode not implemented yet. Given: {path}")


def launch_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Procedural Memory Benchmark")
    parser.add_argument("--config-file", type=str)
    args = parser.parse_args()

    if args.config_file:
        config_mode(args.config_file)
    else:
        interactive_mode()
