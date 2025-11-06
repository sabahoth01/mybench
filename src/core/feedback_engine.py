class ManualFeedback:
    def provide(self, step, model_output):
        return input("Enter feedback manually: ")

class AutoFeedback:
    def provide(self, step, model_output):
        # trivial rule-based for now
        if any(word in model_output.lower() for word in ["done", "transfer", "completed"]):
            return "Correct, proceed."
        else:
            return "Incorrect or incomplete."

def get_feedback_provider(mode="manual"):
    if mode == "auto":
        return AutoFeedback()
    return ManualFeedback()
