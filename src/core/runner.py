import json
from datetime import datetime
from pathlib import Path
from memory_module.memory_manager import MemoryManager

TRIAL_LOG = Path("outputs/logs/trials.jsonl")

def run_trial_series(task, n_trials, feedback_provider, perturb_type="none"):
    """Run multiple trials on the same task, logging episodic and procedural memory."""
    print(f"Running {n_trials} trials for task {task['task_id']}")
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)

    memory = MemoryManager()
    signature = task.get("signature", f"{task['axis']}_{task['level']}")

    for t in range(1, n_trials + 1):
        print(f"\n Trial {t}/{n_trials} ")
        result = run_single_trial(task, feedback_provider)
        success = evaluate_trial_success(result)

        # Append to trial logs 
        memory.after_trial(
            task_id=task["task_id"],
            trial_idx=t,
            task_class=task["axis"],
            signature=signature,
            input_data={"task_meta": task},
            model_plan=[s["instruction"] for s in task["steps"]],
            verifier={"overall": int(success)},
            feedback=[s["feedback"] for s in result["steps"]],
            success=success
        )

        # Also log trial to human-readable file
        with open(TRIAL_LOG, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    print("\nAll trials completed.")
    print("Current procedural memory entries:")
    for p in memory.procedural.procedures:
        print(json.dumps(p, indent=2, ensure_ascii=False))


def run_single_trial(task, feedback_provider):
    """Execute one trial of the task interactively."""
    trial_data = {
        "task_id": task["task_id"],
        "axis": task["axis"],
        "level": task["level"],
        "trial_timestamp": datetime.now().isoformat(),
        "steps": []
    }

    for i, step in enumerate(task["steps"], start=1):
        print(f"\nStep {i}: {step['instruction']}")
        user_output = input("Model output (simulate): ").strip()
        feedback = feedback_provider.provide(step, user_output)
        print(f"Feedback: {feedback}")

        trial_data["steps"].append({
            "step_index": i,
            "instruction": step["instruction"],
            "model_output": user_output,
            "feedback": feedback
        })

    print("Trial completed.")
    return trial_data


def evaluate_trial_success(trial_data):
    """
    Very simple success rule: all steps must have 'correct' in feedback
    or any custom condition you define.
    """
    feedbacks = [s["feedback"] for s in trial_data["steps"]]
    return all("correct" in f.lower() or "good" in f.lower() for f in feedbacks)
