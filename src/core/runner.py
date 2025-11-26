# import json
# from datetime import datetime
# from pathlib import Path
# from memory_module import memory_manager
# from src.core.feedback_engine import FeedbackEngine
# from src.core.pertubation_engine import PerturbationEngine

# TRIAL_LOG = Path("outputs/logs/trials.jsonl")


# def run_trial_series(task, n_trials, feedback_policy="minimal", perturb_policy="none"):
#     """
#     Run multiple trials with feedback and perturbation policies applied.
#     """
#     TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
#     memory = memory_manager()
#     feedback_engine = FeedbackEngine(policy_name=feedback_policy)
#     perturb_engine = PerturbationEngine(policy_name=perturb_policy)

#     print(f"\nRunning {n_trials} trials for task {task['task_id']}")
#     metrics = {"success_rate": 0.0}

#     for trial_idx in range(1, n_trials + 1):
#         print(f"\n--- Trial {trial_idx}/{n_trials} ---")

#         # Apply perturbation
#         perturbed_task = perturb_engine.apply(task.copy(), metrics)

#         trial_result = run_single_trial(perturbed_task)
#         success = evaluate_trial_success(trial_result)
#         trial_result["success"] = success

#         # Feedback
#         feedback = feedback_engine.give_feedback(trial_result)
#         trial_result["policy_feedback"] = feedback

#         # Update metrics// not yet tested.. must be correctdy ... simple test
#         metrics["success_rate"] = ((metrics["success_rate"] * (trial_idx - 1)) + int(success)) / trial_idx

#         # Update memory// not yet tested must be corrected according to the methodology file
#         memory.after_trial(
#             task_id=task["task_id"],
#             trial_idx=trial_idx,
#             task_class=task.get("category_key"),
#             signature=f"{task.get('category_key')}_{task.get('category_name')}",
#             input_data={"task_meta": task},
#             model_plan=[s["instruction"] for s in trial_result["steps"]],
#             verifier={"overall": int(success)},
#             feedback=[s.get("feedback", "") for s in trial_result["steps"]],
#             success=success,
#         )

#         # Log trial
#         with open(TRIAL_LOG, "a", encoding="utf-8") as f:
#             json.dump(trial_result, f, ensure_ascii=False)
#             f.write("\n")

#     print("\nAll trials completed.")
#     print("Procedural memory updated.")


# def run_single_trial(task):
#     """
#     Run one trial interactively.
#     """
#     trial_data = {
#         "task_id": task["task_id"],
#         "category_key": task["category_key"],
#         "category_name": task["category_name"],
#         "timestamp": datetime.now().isoformat(),
#         "steps": []
#     }

#     for i, step in enumerate(task["steps"], start=1):
#         print(f"\nStep {i}: {step['instruction']}")
#         model_output = input("Model output (simulate): ").strip()

#         # Simulate step result // not final just basic conception
#         passed = "correct" in model_output.lower() or "done" in model_output.lower()
#         trial_data["steps"].append({
#             "index": i,
#             "instruction": step["instruction"],
#             "model_output": model_output,
#             "passed": passed
#         })

#     return trial_data
# ## must be corrected this is just squelette
# def evaluate_trial_success(trial_result):
#     """Success if all steps passed."""
#     return all(step["passed"] for step in trial_result["steps"])

import json
from datetime import datetime
from pathlib import Path

from memory_module import memory_manager
from src.core.feedback_engine import FeedbackEngine
from src.core.pertubation_engine import PerturbationEngine

TRIAL_LOG = Path("outputs/logs/trials.jsonl")


def run_trial_series(task, n_trials, feedback_policy="minimal", perturb_policy="none"):
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)

    memory = memory_manager()
    feedback_engine = FeedbackEngine(policy_name=feedback_policy)
    perturb_engine = PerturbationEngine(policy_name=perturb_policy)

    metrics = {"success_rate": 0.0}

    print(f"\nRunning {n_trials} trials for task {task['task_id']}")

    for trial_idx in range(1, n_trials + 1):
        print(f"\n--- Trial {trial_idx}/{n_trials} ---")

        perturbed_task = perturb_engine.apply(dict(task), metrics)

        trial = run_single_trial(perturbed_task)
        success = evaluate_trial_success(trial)
        trial["success"] = success

        feedback = feedback_engine.give_feedback(trial)
        trial["policy_feedback"] = feedback

        metrics["success_rate"] = (
            (metrics["success_rate"] * (trial_idx - 1)) + int(success)
        ) / trial_idx

        memory.after_trial(
            task_id=task["task_id"],
            trial_idx=trial_idx,
            task_class=task.get("category_key"),
            signature=f"{task.get('category_key')}_{task.get('category_name')}",
            input_data={"task_meta": task},
            model_plan=[s["instruction"] for s in trial["steps"]],
            verifier={"overall": int(success)},
            feedback=[s.get("feedback", "") for s in trial["steps"]],
            success=success
        )

        with open(TRIAL_LOG, "a", encoding="utf-8") as f:
            json.dump(trial, f, ensure_ascii=False)
            f.write("\n")

    print("\nAll trials completed.")


def run_single_trial(task):
    data = {
        "task_id": task["task_id"],
        "category_key": task["category_key"],
        "category_name": task["category_name"],
        "timestamp": datetime.now().isoformat(),
        "steps": []
    }

    for idx, step in enumerate(task["steps"], start=1):
        instruction = step.get("action") or step.get("instruction")
        print(f"\nStep {idx}: {instruction}")
        model_output = input("Model output (simulate): ").strip()

        passed = any(tok in model_output.lower() for tok in ("correct", "done"))

        data["steps"].append({
            "index": idx,
            "instruction": instruction,
            "model_output": model_output,
            "passed": passed
        })

    return data


def evaluate_trial_success(trial):
    return all(s["passed"] for s in trial["steps"])
