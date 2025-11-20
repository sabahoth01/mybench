import json
from pathlib import Path
from src.generator.task_generator import generate_multiple_instances
from src.core.runner import run_trial_series

TEST_CATEGORY_FILE = Path("configs/test_category.json")
with open(TEST_CATEGORY_FILE, "r", encoding="utf-8") as f:
    TEST_CATEGORIES = json.load(f)
ORDERED_CATEGORIES = [
    (i, key, info) 
    for i, (key, info) in enumerate(TEST_CATEGORIES.items(), start=1)]

def interactive_mode():
    print("\n Procedural Memory Benchmark\n")

    print("\nAvailable task categories:\n")
    for num, key, info in ORDERED_CATEGORIES:
        print(f"{num}. {info['name']} ({key}): {info['description']}")
    try:
        selection = int(input("\nEnter category number: \n\t-> ").strip())
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if not (1 <= selection <= len(ORDERED_CATEGORIES)):
        print("Invalid category number.")
        return

    num, category_key, category_info = ORDERED_CATEGORIES[selection - 1]
    category_name = category_info["name"]
    print(f"\nSelected category: {category_name} ({category_key})")

    task_prompt = input(
        f"\nEnter base task prompt for '{category_name}'\n"
        "For example: Transfer 50 rubles from Bob to Ann...\n\t-> "
    ).strip()

    n_instances = int(input("\nHow many task instances to generate?\n\t-> ").strip())
    n_trials = int(input("\nHow many trials per instance?\n\t-> ").strip())

    feedback_policy = input(
        "\nSelect feedback policy (minimal, descriptive, stepwise, applied): \n\t-> "
    ).strip()

    perturb_policy = input(
        "\nSelect perturbation policy (none, probalistic, performance_based):\n\t->  "
    ).strip()

    print(f"\nGenerating {n_instances} tasks for category '{category_name}'...")
    tasks = generate_multiple_instances(
        prompt=task_prompt,
        category_key=category_key,
        category_name=category_name,
        n_instances=n_instances
    )
    print(f"\nGenerated {len(tasks)} tasks.")

    start = input("\nStart trials now? (y/n): \n\t-> ").strip().lower()
    if start not in ("y", "yes"):
        print("\nExiting without running trials.")
        return

    for i, task in enumerate(tasks, start=1):
        print(f"\n Task Instance {i}/{len(tasks)} ({task['task_id']}) ")
        run_trial_series(
            task,
            n_trials,
            feedback_policy=feedback_policy,
            perturb_policy=perturb_policy
        )

def config_mode(config):
    print(f"Running benchmark from config file: {config}")
    # TODO: YAML config-driven batch mode

def launch_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Procedural Memory Benchmark")
    parser.add_argument("--config-file", type=str, help="YAML config file for batch mode.")
    args = parser.parse_args()

    if args.config_file:
        config_mode(args.config_file)
    else:
        interactive_mode()
