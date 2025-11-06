import argparse
from src.generator.task_generator import generate_from_prompt
from src.core.runner import run_trial_series
from src.core.feedback_engine import get_feedback_provider

def interactive_mode():
    print("\nProcedural Memory Benchmark — Interactive Mode")
    task_prompt = input("Enter base task (e.g. 'Transfer funds between accounts'): ").strip()

    axis = "C1"  # fixed for now
    level = input("Choose sublevel [1.1 / 1.2]: ").strip().upper()
    n_instances = int(input("How many task instances to generate? ").strip())
    n_trials = int(input("How many trials per instance? ").strip())

    feedback_type = input("Feedback policy [manual / auto(LLM as a judge)]: ").strip().lower()
    perturb_type = "none"  # placeholder for now

    print(f"\nGenerating {n_instances} task instances for class {axis}.{level}...")
    tasks = [generate_from_prompt(task_prompt, axis, f"{axis}.{level}") for _ in range(n_instances)]

    print(f"Generated {len(tasks)} tasks.")
    start = input("Start trials now? (y/n): ").strip().lower()
    if start != "y":
        print("Exiting without running trials.")
        return

    feedback_provider = get_feedback_provider(feedback_type)
    for i, task in enumerate(tasks, start=1):
        print(f"\n=== Task Instance {i}/{len(tasks)} ({task['task_id']}) ===")
        run_trial_series(task, n_trials, feedback_provider, perturb_type)


def config_mode(config):
    print(f"Running in config mode with: {config}")
    # placeholder: would parse YAML config for batch runs


def launch_cli():
    parser = argparse.ArgumentParser(description="Procedural Memory Benchmark CLI")
    parser.add_argument("--config-file", type=str, help="YAML config file for batch mode.")
    args = parser.parse_args()

    if args.config_file:
        config_mode(args.config_file)
    else:
        interactive_mode()
