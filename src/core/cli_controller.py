import json
from pathlib import Path
from src.generator.task_generator import generate_from_prompt
from src.core.runner import run_trial_series

TASK_CATEGORY_FILE = Path("configs/task_category.json")

# Load categories from configs/test_category
with open(TASK_CATEGORY_FILE, "r", encoding="utf-8") as f:
    TASK_CATEGORIES = json.load(f)

# Reverse lookup: name → key
NAME_TO_KEY = {v["name"].lower(): k for k, v in TASK_CATEGORIES.items()}


def interactive_mode():
    print("\n Procedural Memory Benchmark\n\n\n")

    # Show available categories
    print("\nAvailable task categories:\n")
    for key, info in TASK_CATEGORIES.items():
        print(f"- {info['name']} ({key}): {info['description']}")

    # User selects category by name
    category_name = input("\nEnter test category name correctly as written: \n\t-> ").strip().lower()
    if category_name not in NAME_TO_KEY:
        print(f"Unknown category '{category_name}'!")
        return

    category_key = NAME_TO_KEY[category_name]

    task_prompt = input(f"\nEnter base task prompt for '{category_name}'\nFor example: Transfert 50 rubles from bob to ann...\n\t-> ").strip()
    n_instances = int(input("\nHow many task instances to generate?\n\t-> ").strip())
    n_trials = int(input("\nHow many trials per instance?\n\t-> ").strip())

    # Feedback and perturbation policy
    feedback_policy = input("\nSelect feedback policy (minimal, descriptive, stepwise, applied): \n\t-> ").strip()
    perturb_policy = input("\nSelect perturbation policy (none, probalistic, performance_based):\n\t->  ").strip()

    # Generate tasks
    print(f"\nGenerating {n_instances} tasks for category '{category_name}'...")
    tasks = [generate_from_prompt(task_prompt, category_key, category_name) for _ in range(n_instances)]
    print(f"\nGenerated {len(tasks)} tasks.")

    # Start trials?
    start = input("\nStart trials now? (y/n): \n\t-> ").strip().lower()
    if start != "y" or "Y":
        print("\nExiting without running trials.")
        return

    for i, task in enumerate(tasks, start=1):
        print(f"\n******* Task Instance {i}/{len(tasks)} ({task['task_id']}) *****")
        run_trial_series(task, n_trials, feedback_policy=feedback_policy, perturb_policy=perturb_policy)


def config_mode(config):
    print(f"Running benchmark from config file: {config}")
    # TODO: YAML config-driven batch mode


def launch_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Procedural Memory Benchmark CLI")
    parser.add_argument("--config-file", type=str, help="YAML config file for batch mode.")
    args = parser.parse_args()

    if args.config_file:
        config_mode(args.config_file)
    else:
        interactive_mode()
