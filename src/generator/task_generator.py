import os
import json
import random
import requests
import uuid
import re

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from src.core.constraint_engine import ConstraintEngine
from src.core.task_validator import TaskValidator

load_dotenv()

GENERATED_DIR = Path("data/generated_tasks")
REGISTRY_PATH = Path("data/task_registry.json")
CATEGORY_FILE = Path("configs/test_category.json")
DOMAIN_FILE = Path("configs/task_domaine.json")
KEYWORD_FILE = Path("configs/domain_keywords.json")
CONSTRAINT_FILE = Path("configs/domaine_constraint.json")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _load_json(path: Path):
    if not path.exists():
        if path == CONSTRAINT_FILE: return {}
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_categories(): return _load_json(CATEGORY_FILE)
def _load_domain(): return _load_json(DOMAIN_FILE)
def _load_keywords(): return _load_json(KEYWORD_FILE)
def _load_constraints(): return _load_json(CONSTRAINT_FILE) if CONSTRAINT_FILE.exists() else {}

def _generate_task_id(category_key: str) -> str:
    return f"{category_key}_{uuid.uuid4().hex[:6]}"

def _sample_complexity(category_key: str, categories: dict):
    if category_key not in categories:
        raise ValueError(f"Unknown category {category_key}")
    cx = categories[category_key].get("complexity", {})
    sampled = {k: random.choice(v) if isinstance(v, list) else v for k, v in cx.items()}
    return sampled

def _resolve_domain_info(prompt: str, domain_data: dict):
    prompt_lower = prompt.lower()
    keyword_map = _load_keywords()
    
    # Keyword Map Lookup
    for subtask_name, keywords in keyword_map.items():
        if any(kw in prompt_lower for kw in keywords):
            for domain, sections in domain_data.items():
                for section, tasks in sections.items():
                    if subtask_name in tasks:
                        return domain, section, subtask_name, tasks[subtask_name]

    # Fuzzy Matching on Subtask Names
    for domain_name, domain_sections in domain_data.items():
        for section, subtasks in domain_sections.items():
            for subtask_name, schema_info in subtasks.items():
                clean_name = subtask_name.replace("_", " ")
                if clean_name in prompt_lower:
                     return domain_name, section, subtask_name, schema_info
                     
    return "banking", "general", "unspecified_task", {"inputs": [], "outputs": [], "skills": []}

def _build_extraction_system_prompt(domain, subtask, input_schema, complexity):
    """
    Builds a prompt that forces the agent to extract specific keys defined in task_domaine.json
    and generate simple steps.
    """
    # Create a simplified list of keys for the LLM to focus on
    if isinstance(input_schema, dict):
        keys = list(input_schema.keys())
    elif isinstance(input_schema, list):
        keys = input_schema
    else:
        keys = []
    
    steps_count = complexity.get("procedure_length", 5)

    prompt = f"""
    You are a Procedural Memory Task Generator.
    Your goal is to analyze the user prompt and generate a valid JSON object.

    ### Context
    Domain: {domain}
    Subtask: {subtask}

    ### Instruction
    1. **Extract Inputs**: Look for the following parameters in the user prompt: {json.dumps(keys)}.
    - If a value is present in the prompt, extract it.
    - If a value is missing, set it to unknown as define by the constraint file: unknown or take the default value.
    - Do NOT invent data.
    2. **Generate Steps**: detailed steps to perform this subtask (approx {steps_count} steps).

    ### Response Format
    Return ONLY valid JSON:
    {{
    "inputs": {{ "key": "value" }},
    "steps": [ "step 1", "step 2" ]
    }}
    """
    return prompt

def _llm_generate_and_extract(prompt, system_prompt):
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.45, # Low temperature for extraction accuracy to prevent creativity, invent etc...
        "max_tokens": 2000
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        
        # Robust JSON extraction
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1: raise ValueError("No JSON in response")
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"LLM Error: {e}")
        # Fallback to prevent crash
        return {"inputs": {}, "steps": ["Error generating steps"]}

def _synthesize_outputs(c_engine, sub_rules, validated_inputs):
    """
    Generates synthetic output data (IDs, receipts) based on constraint rules.
    """
    synthesized = {}
    output_rules = sub_rules.get("outputs", {})

    if "account_id" in output_rules:
        synthesized["account_id"] = c_engine.generate_account_id(validated_inputs)
    
    if "transaction_id" in output_rules:
        synthesized["transaction_id"] = c_engine.generate_transaction_id()

    # Format Templates (Receipts)
    final_outputs = c_engine.apply_output_templates(sub_rules, validated_inputs, synthesized)
    
    return final_outputs

def generate_from_prompt(prompt: str, category_key: str, category_name: str):
    random.seed()
    categories = _load_categories()
    domain_data = _load_domain()
    constraints = _load_constraints()
    complexity = _sample_complexity(category_key, categories)
    
    # Initialize Engines
    c_engine = ConstraintEngine(constraints)
    validator = TaskValidator()

    # 1. Resolve Domain & Schema
    domain_name, section, subtask, schema_info = _resolve_domain_info(prompt, domain_data)
    
    # Get strict rules from domaine_constraint.json
    sub_rules = c_engine.get_subtask_constraints(domain_name, section, subtask)
    
    # Merge basic schema from task_domaine with strict rules
    input_keys = schema_info.get("inputs", [])

    # for LLM Extraction & Generation,, We use a custom system prompt here to force Extraction
    system_prompt = _build_extraction_system_prompt(domain_name, subtask, input_keys, complexity)
    llm_result = _llm_generate_and_extract(prompt, system_prompt)
    
    raw_inputs = llm_result.get("inputs", {})
    raw_steps = llm_result.get("steps", [])

    # Apply Input Constraints (Fill Defaults / Handle Missing)
    validated_inputs = c_engine.apply_input_rules(sub_rules, raw_inputs)

    # Synthesize Outputs
    # This generates the Account IDs and formatting receipts using Python
    outputs = _synthesize_outputs(c_engine, sub_rules, validated_inputs)

    # Validate Steps
    try:
        v_res = validator.validate_steps({"steps": raw_steps})
        steps = v_res["steps"]
    except Exception as e:
        print(f"Validation warning: {e}")
        # Fallback to string conversion if validation fails but we have data
        steps = [str(s) for s in raw_steps]

    # Build Final Object
    task = {
        "task_id": _generate_task_id(category_key),
        "category_key": category_key,
        "category_name": category_name,
        "domain": domain_name,
        "section": section,
        "subtask": subtask,
        "steps": steps,
        "inputs": validated_inputs,
        "outputs": outputs,
        "skills": schema_info.get("skills", []),
        "complexity": complexity,
        "prompt": prompt,
        "created_at": datetime.now().isoformat()
    }

    # Save
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATED_DIR / f"{task['task_id']}.json", "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    # Registry Update
    if REGISTRY_PATH.exists():
        registry = _load_json(REGISTRY_PATH)
    else:
        registry = []
    
    # Avoid duplicate IDs in registry if running fast loop
    if not any(r['task_id'] == task['task_id'] for r in registry):
        registry.append({
            "task_id": task["task_id"],
            "category_key": category_key,
            "category_name": category_name
        })
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    return task

def generate_multiple_instances(prompt, category_key, category_name, n_instances=2):
    tasks = []
    attempts = 0
    while len(tasks) < n_instances and attempts < n_instances * 3:
        attempts += 1
        try:
            t = generate_from_prompt(prompt, category_key, category_name)
            tasks.append(t)
        except Exception as e:
            print(f"Generation failed: {e}")
    return tasks