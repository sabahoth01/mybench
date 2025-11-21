import json

def build_prompt(
    task_prompt: str,
    category_key: str,
    category_name: str,
    domain_name: str,
    section: str,
    subtask: str,
    catalog_schema: dict,
    constraints: dict,
    complexity: dict
):
    sc = constraints.get(category_key, {}).get(section, {}).get(subtask, {})

    return f"""
      You are a deterministic procedural step generator.

      CONTEXT:
      - Category: {category_name} ({category_key})
      - Domain: {domain_name}
      - Section: {section}
      - Subtask: {subtask}
      - User prompt: {task_prompt}

      SCHEMA:
      - Inputs: {catalog_schema.get('inputs', [])}
      - Outputs: {catalog_schema.get('outputs', [])}
      - Skills: {catalog_schema.get('skills', [])}

      CONSTRAINTS:
      {json.dumps(sc, indent=2)}

      RULES:
      1. Output JSON only.
      2. JSON must contain exactly:
        {{
          "steps": [
            "Step 1 instruction",
            "Step 2 instruction",
            ...
          ]
        }}
      3. Steps are short imperative sentences.
      4. Do NOT invent input values; reference only inputs provided.
      5. Do NOT include text outside the JSON.
      6. Each step should be complete and sequential.

      START.
      """.strip()
