import random
import yaml
# must be corrected, this is just a simple test part that i have tried
class FeedbackEngine:
    def __init__(self, policy_name="minimal", config_path="configs/feedback_policies.yaml"):
        with open(config_path) as f:
            self.policies = yaml.safe_load(f)["feedback_policies"]
        self.policy = self.policies[policy_name]

    def give_feedback(self, trial_result):
        """Return feedback based on the configured policy."""
        level = self.policy.get("level")
        style = self.policy.get("style")

        if level == "trial":
            return self._trial_feedback(trial_result)
        elif level == "step":
            return self._step_feedback(trial_result)
        else:
            raise ValueError("Invalid feedback level")

    def _trial_feedback(self, result):
        if result["success"]:
            return self.policy["content"]["success"]
        return self.policy["content"]["failure"]

    def _step_feedback(self, result):
        fb = []
        for i, step in enumerate(result["steps"], start=1):
            if step["passed"]:
                fb.append(self.policy["content"].get("pass", "Step OK"))
            else:
                msg = self.policy["content"].get("fail", "Step failed")
                if self.policy["style"] == "interactive":
                    prompt = random.choice(self.policy["content"]["prompt_templates"])
                    msg += " " + prompt.format(n=i)
                fb.append(msg)
        return fb