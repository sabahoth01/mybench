import random
import yaml
# must be corrected to take into account the complexity leveler of each block that i have defined
class PerturbationEngine:
    def __init__(self, policy_name="none", config_path="configs/perturbation_policies.yaml"):
        with open(config_path) as f:
            self.policies = yaml.safe_load(f)["perturbation_policies"]
        self.policy = self.policies[policy_name]

    def apply(self, task_instance, metrics):
        """Apply perturbations according to policy."""
        rules = self.policy.get("rules", [])
        for rule in rules:
            if "probability" in rule and random.random() < rule["probability"]:
                self._apply_rule(task_instance, rule["type"])
            elif "condition" in rule and self._evaluate_condition(rule["condition"], metrics):
                for action in rule.get("actions", []):
                    self._apply_rule(task_instance, action)
        return task_instance

    def _apply_rule(self, task, rule_type):
        # Example perturbations
        if rule_type == "increase_difficulty":
            task["complexity_level"] = min(task.get("complexity_level", 1) + 1, 5)
        elif rule_type == "add_constraint":
            task.setdefault("constraints", []).append("new_constraint")
        elif rule_type == "simplify_inputs":
            task["inputs"] = task["inputs"][:max(1, len(task["inputs"]) - 1)]
        elif rule_type == "tool_unavailability":
            task["toolset"] = task.get("toolset", [])
            if task["toolset"]:
                removed = random.choice(task["toolset"])
                task["toolset"].remove(removed)
                task.setdefault("perturbation_notes", []).append(f"Tool {removed} unavailable.")
        elif rule_type == "input_noise":
            task.setdefault("noise", []).append("Random value perturbation.")
        return task

    def _evaluate_condition(self, expr, metrics):
        """Safe eval for conditions like 'success_rate > 0.8'."""
        try:
            return eval(expr, {}, metrics)
        except Exception:
            return False