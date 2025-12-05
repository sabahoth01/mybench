import random
import yaml
from pathlib import Path

class PerturbationEngine:
    def __init__(self, policy_name="none",
                 policy_path="configs/perturbation_policy.yaml",
                 resources_path="configs/perturbation_resources.yaml"):

        p_path = Path(policy_path)
        r_path = Path(resources_path)

        if p_path.exists():
            with open(p_path, 'r') as f:
                self.policies = yaml.safe_load(f).get("perturbation_policies", {})
        else:
            self.policies = {"none": {"mode": "off"}}

        if r_path.exists():
            with open(r_path, 'r') as f:
                self.resources = yaml.safe_load(f).get("perturbation_resources", {})
        else:
            self.resources = {}

    
        if policy_name in self.policies:
            self.policy = self.policies[policy_name]
        else:
            print(f"WARNING: Policy '{policy_name}' not found. Defaulting to 'none'.")
            self.policy = self.policies.get("none", {"mode": "off"})
        
        self.incorrect_streak = 0

    def apply(self, task, metrics):
        mode = self.policy.get("mode", "off")

        if mode == "off":
            return task
        elif mode == "probabilistic":
            return self._apply_probabilistic(task)
        elif mode == "progressive":
            return self._apply_progressive(task, metrics)
        elif mode == "adaptive":
            return self._apply_adaptive(task, metrics)
        
        return task

    def _apply_probabilistic(self, task):
    
        rules = self.policy.get("rules", {})
        for rule_type, prob in rules.items():
            if random.random() < prob:
                self._execute(task, rule_type)
        return task

    def _apply_progressive(self, task, metrics):
        trial_num = metrics.get("trial_number", 1) # Note: Runner passes trial_idx via metrics if needed, or we rely on loop
        # Usually metrics['success_rate'] is passed. For progressive, we need a counter.
        
        interval = self.policy.get("step_interval", 3)
        # Using a default counter if not provided
        t_num = metrics.get("trial_number", 1) 

        if t_num % interval == 0:
            for action in self.policy.get("actions", []):
                self._execute(task, action)
        return task

    def _apply_adaptive(self, task, metrics):
        sr = metrics.get("success_rate", 0)
        thresholds = self.policy.get("thresholds", {})
        
        if sr > thresholds.get("high_success", 0.8):
            for action in self.policy.get("high_success_actions", []):
                self._execute(task, action)
        elif sr < thresholds.get("low_success", 0.3):
            for action in self.policy.get("low_success_actions", []):
                self._execute(task, action)
        return task

    def _execute(self, task, action):
        
        if action == "tool_unavailability":
            pool = task.get("toolset", [])
            if pool:
                removed = random.choice(pool)
                if removed in pool:
                    pool.remove(removed)
                    task.setdefault("notes", []).append(f"Tool removed: {removed}")
        
        elif action == "step_shuffle":
            if "steps" in task and isinstance(task["steps"], list):
                random.shuffle(task["steps"])
                task.setdefault("notes", []).append("Steps shuffled")

        elif action == "input_noise":
             options = self.resources.get("input_noise", {}).get("ambiguity", ["Noise"])
             task.setdefault("input_noise", []).append(random.choice(options))