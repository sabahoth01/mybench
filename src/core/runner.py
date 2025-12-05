import json
import copy
from datetime import datetime
from pathlib import Path
from src.core.feedback_engine import FeedbackEngine
from src.core.pertubation_engine import PerturbationEngine
from memory_module.memory_manager import MemoryManager

memory = MemoryManager()

TRIAL_LOG = Path("outputs/logs/trials.jsonl")

def run_trial_series(task, n_trials, feedback_policy="binary_step", perturb_policy="none"):
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Engines
    feedback_engine = FeedbackEngine(policy_name=feedback_policy)
    perturb_engine = PerturbationEngine(policy_name=perturb_policy)

    metrics = {"success_rate": 0.0}

    print(f"\n=== Running {n_trials} trials for task {task['task_id']} ===")
    print(f"Policies: Feedback='{feedback_policy}', Perturbation='{perturb_policy}'")

    for trial_idx in range(1, n_trials + 1):
        print(f"\n--- Trial {trial_idx}/{n_trials} ---")

        # Deep copy to avoid perturbation contamination
        task_instance = copy.deepcopy(task)

        # Apply perturbation
        perturbed_task = perturb_engine.apply(task_instance, metrics)

        # Run the trial
        trial_result = run_single_trial(perturbed_task, feedback_engine, trial_idx)

        # Evaluate trial success
        success = evaluate_trial_success(trial_result)
        trial_result["success"] = success

        # Trial-level feedback policy
        if feedback_policy == "binary_trial":
            fb_data = feedback_engine.request_feedback(trial=True)
            trial_result["trial_feedback"] = "correct" if fb_data["passed"] else "incorrect"
        else:
            trial_result["trial_feedback"] = None

        # Update cumulative average success rate
        metrics["success_rate"] = (
            (metrics["success_rate"] * (trial_idx - 1)) + int(success)
        ) / trial_idx

        # Procedural Memory Update
        memory.after_trial(
            task_id=task["task_id"],
            trial_idx=trial_idx,
            test_category=f"{task.get('category_key')}_{task.get('category_name')}",
            input_data={"task_meta": task},
            model_plan=[s["instruction"] for s in trial_result["steps"]],
            verifier={"overall": int(success)},
            feedback=[s.get("feedback") for s in trial_result["steps"]],
            success=success,
        )

        # Write trial log
        with open(TRIAL_LOG, "a", encoding="utf-8") as f:
            json.dump(trial_result, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nAll trials completed. Final Success Rate: {metrics['success_rate']:.2f}")


# in case we just have one trial to run
def run_single_trial(task, feedback_engine, trial_idx):
    trial_data = {
        "task_id": task["task_id"],
        "trial_index": trial_idx,
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "perturbations_applied": task.get("notes", []) + task.get("input_noise", [])
    }

    steps = task.get("steps", [])
    if not steps:
        print("Error: No steps found in task.")
        return trial_data

    for idx, step in enumerate(steps, start=1):

        # Normalize instruction text
        instruction = step if isinstance(step, str) else step.get("instruction", str(step))
        print(f"\nStep {idx}: {instruction}")

        # FEEDBACK ENGINE handles full evaluation
        feedback_data = feedback_engine.request_feedback(step_number=idx, trial=False)

        # Generate system message based on YAML templates
        system_response = feedback_engine.generate_feedback_message(
            passed=feedback_data["passed"],
            step_number=idx
        )
        print(f"System: {system_response}")

        # Save step results
        trial_data["steps"].append({
            "index": idx,
            "instruction": instruction,
            "passed": feedback_data["passed"],
            "feedback": "correct" if feedback_data["passed"] else "incorrect",
            "hint": feedback_data["hint"]
        })

    return trial_data


# to be arranged :))
def evaluate_trial_success(trial):
    if not trial["steps"]:
        return False
    return all(step["passed"] for step in trial["steps"])
