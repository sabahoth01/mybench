import yaml

class FeedbackEngine:
    def __init__(self, policy_name="binary_step", config_path="configs/feedback_policy.yaml"):
        with open(config_path) as f:
            self.policies = yaml.safe_load(f)["feedback_policies"]

        if policy_name not in self.policies:
            raise ValueError(f"Unknown feedback policy: {policy_name}")

        self.policy = self.policies[policy_name]

        # Tracks consecutive incorrect attempts for each step/trial
        self.incorrect_counter = 0

    def request_feedback(self, step_number=None, trial=False):
        # If binary_trial is active, we might skip step feedback logic 
        # depending on your design. Assuming we use it for both levels:
        
        if trial:
            prompt = "\n[Trial End] Was the whole trial correct? (correct/incorrect): "
        else:
            prompt = f"\n[Step {step_number}] Correct? (correct/incorrect): "

        while True:
            fb = input(prompt).strip().lower()
            if fb in ["correct", "c", "1", "yes", "y"]:
                self.incorrect_counter = 0
                return {"passed": True, "hint": None}
            
            elif fb in ["incorrect", "i", "0", "no", "n"]:
                self.incorrect_counter += 1
                
                # Check threshold
                if self.incorrect_counter >= 4:
                    print(">> Repeated failures detected. Please provide a hint.")
                    hint = input(">> Hint: ").strip()
                    self.incorrect_counter = 0 # Reset after hint
                    return {"passed": False, "hint": hint}
                
                return {"passed": False, "hint": None}
            
            else:
                print("Invalid input. Type 'correct' or 'incorrect'.")

    def generate_feedback_message(self, passed, step_number=None):
        """Returns the message defined in YAML for the policy."""
        content = self.policy["content"]
        key = "correct" if passed else "incorrect"
        msg = content[key]

        if "{n}" in msg and step_number is not None:
            msg = msg.format(n=step_number)

        return msg
