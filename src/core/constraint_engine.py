import re
import secrets
import logging
from datetime import date

logging.basicConfig(level=logging.WARNING)

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
        Fill in input values:
        - Use user provided values if present.
        - Use 'default' if optional and missing.
        - Use 'unknown' only for required missing fields.
        """
        processed = {}
        input_rules = subtask_rules.get("inputs", {})
        unknown = subtask_rules.get("unknown_value", "unknown")

        for key, rule in input_rules.items():
            if key in user_inputs and user_inputs[key] is not None:
                processed[key] = user_inputs[key]
            else:
                if rule.get("required", False):
                    processed[key] = rule.get("unknown_value", unknown)
                else:
                    # Optional field: prefer default, else unknown
                    processed[key] = rule.get("default", unknown)
        return processed

    def generate_account_id(self, customer_data):
        """
        Format: FIRSTINITIAL-6hex (uppercase initial)
        If no first name present -> use 'X' as initial.
        """
        initial = "X"
        if isinstance(customer_data, dict):
            fn = (
                customer_data.get("first_name")
                or customer_data.get("firstname")
                or customer_data.get("name")
            )
            if isinstance(fn, str) and fn.strip():
                initial = fn.strip()[0].upper()

        h = secrets.token_hex(3)
        return f"{initial}-{h}"

    def generate_transaction_id(self):
        return f"tsc-{secrets.token_hex(3)}"

    def apply_output_templates(self, subtask_rules: dict, inputs: dict, outputs: dict):
        """
        Apply format_template for outputs.
        Logs a warning if formatting fails.
        """
        templates = subtask_rules.get("outputs", {})
        today = date.today().isoformat()
        result = dict(outputs)

        for field, rule in templates.items():
            if "format_template" in rule:
                template = rule["format_template"]
                mapping = {}
                mapping.update(inputs)
                mapping.update(result)
                mapping["date"] = today
                try:
                    result[field] = template.format(**mapping)
                except Exception as e:
                    logging.warning(f"Output template formatting failed for field '{field}': {e}")
                    result[field] = template.replace("{date}", today)
        return result

    def validate_allowed(self, value, rule):
        allowed = rule.get("allowed")
        unknown = rule.get("unknown_value", "unknown")
        if allowed:
            if value in allowed:
                return value
            return unknown
        return value

    def validate_regex(self, value, pattern: str):
        """
        Uses fullmatch for exact format validation (e.g., ID patterns).
        Returns True if value matches the entire pattern.
        """
        if not isinstance(value, str):
            return False
        return bool(re.fullmatch(pattern, value))