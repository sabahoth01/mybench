import re
import secrets
from datetime import date

class ConstraintEngine:
    def __init__(self, constraints: dict):
        self.constraints = constraints or {}

    def get_subtask_constraints(self, category, section, subtask):
        return (
            self.constraints
            .get(category, {})
            .get(section, {})
            .get(subtask, {})
        )

    def apply_input_rules(self, subtask_rules: dict, user_inputs: dict):
        """
        Apply defaults and unknowns for inputs according to subtask_rules.
        Returns processed_inputs dict.
        """
        processed = {}
        inputs_rules = subtask_rules.get("inputs", {})
        unknown_token = subtask_rules.get("unknown_value", "unknown")

        for key, rule in inputs_rules.items():
            if key in user_inputs and user_inputs[key] is not None:
                processed[key] = user_inputs[key]
            else:
                if rule.get("required", False):
                    processed[key] = rule.get("unknown_value", unknown_token)
                else:
                    processed[key] = rule.get("default", rule.get("unknown_value", unknown_token))
        return processed

    def generate_account_id(self, customer_data):
        """
        Format: FIRSTINITIAL-6hex (uppercase initial)
        If no first name present -> use 'X' as initial.
        """
        first = "X"
        if isinstance(customer_data, dict):
            fn = customer_data.get("first_name") or customer_data.get("firstname") or customer_data.get("name")
            if isinstance(fn, str) and fn.strip():
                first = fn.strip()[0].upper()
        hex6 = secrets.token_hex(3)  # 3 bytes -> 6 hex chars
        return f"{first}-{hex6}"

    def generate_transaction_id(self):
        hex6 = secrets.token_hex(3)
        return f"tsc-{hex6}"

    def apply_output_templates(self, subtask_rules: dict, inputs: dict, outputs: dict):
        """
        Apply format_template for outputs that have templates.
        Insert date YYYY-MM-DD.
        """
        templates = subtask_rules.get("outputs", {})
        today = date.today().isoformat()
        out = dict(outputs)  # copy

        for field, rule in templates.items():
            if "format_template" in rule:
                template = rule["format_template"]
                # safe format: only known placeholders should exist
                # prepare mapping
                mapping = {}
                mapping.update(inputs)
                mapping.update(out)
                mapping["date"] = today
                try:
                    out[field] = template.format(**mapping)
                except Exception:
                    # fallback to a compact representation
                    out[field] = template.replace("{date}", today)
        return out

    def validate_allowed(self, value, rule):
        """
        If rule specifies allowed list, ensure value is in it;
        otherwise return a default or unknown.
        """
        allowed = rule.get("allowed")
        unknown = rule.get("unknown_value") or "unknown"
        if allowed:
            if value in allowed:
                return value
            # else fallback to first allowed? safer to return unknown
            return unknown
        return value

    def validate_regex(self, value: str, pattern: str) -> bool:
        if not isinstance(value, str):
            return False
        return bool(re.match(pattern, value))
