import logging

class TaskValidator:
    def validate_steps(self, llm_json):
        """
        Validates that the output contains a simple list of string steps.
        Format expected: { "steps": ["Step 1", "Step 2", ...] }
        """
        if not isinstance(llm_json, dict):
            # Attempt to handle if a list was passed directly
            if isinstance(llm_json, list):
                raw_steps = llm_json
            else:
                raise ValueError(f"LLM output must be a dictionary object, got {type(llm_json)}")
        else:
            raw_steps = llm_json.get("steps")
        
        if not isinstance(raw_steps, list):
            raise ValueError("Output must contain a 'steps' key with a list value.")

        cleaned_steps = []
        
        for item in raw_steps:
            # already a string (Desired format)
            if isinstance(item, str):
                s = item.strip()
                if s: 
                    cleaned_steps.append(s)
            
            # The step is a dict (e.g. {"step": "text"}), extract the text
            elif isinstance(item, dict):
                # Find the first string value in the dict
                text = next((v for v in item.values() if isinstance(v, str)), None)
                if text and text.strip():
                    cleaned_steps.append(text.strip())
            
            # Item is something else (int, etc), cast to string
            else:
                s = str(item).strip()
                if s:
                    cleaned_steps.append(s)

        if not cleaned_steps:
            raise ValueError("The 'steps' list is empty or contained no valid text.")

        # Return consistent dictionary format
        return {"steps": cleaned_steps}