import os
import json
import random
import requests
import uuid
import yaml
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

FEEDBACK_POLICY_FILE = Path("configs/feedback_policy.yaml")
PERTURBATION_POLICY_FILE = Path("configs/perturbation_policy.yaml")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# loaders (json from configs)
def _load_json(path: Path):
    if not path.exists():
        if path == CONSTRAINT_FILE:
            return {}
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_categories(): return _load_json(CATEGORY_FILE)
def _load_domain(): return _load_json(DOMAIN_FILE)
def _load_keywords(): return _load_json(KEYWORD_FILE)
def _load_constraints(): return _load_json(CONSTRAINT_FILE) if CONSTRAINT_FILE.exists() else {}

def _load_feedback_policies():
    if not FEEDBACK_POLICY_FILE.exists():
        return {}
    with open(FEEDBACK_POLICY_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_perturbation_policies():
    if not PERTURBATION_POLICY_FILE.exists():
        return {}
    with open(PERTURBATION_POLICY_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_task_id(category_key: str) -> str:
    return f"{category_key}_{uuid.uuid4().hex[:6]}"

def _sample_complexity(category_key: str, categories: dict):
    if category_key not in categories:
        raise ValueError(f"Unknown category {category_key}")
    cx = categories[category_key].get("complexity", {})
    return {k: random.choice(v) if isinstance(v, list) else v for k, v in cx.items()}

def _resolve_domain_info(prompt: str, domain_data: dict):
    prompt_lower = prompt.lower()
    keyword_map = _load_keywords()
    for subtask_name, keywords in keyword_map.items():
        if any(kw in prompt_lower for kw in keywords):
            for domain, sections in domain_data.items():
                for section, tasks in sections.items():
                    if subtask_name in tasks:
                        return domain, section, subtask_name, tasks[subtask_name]
    for domain_name, domain_sections in domain_data.items():
        for section, subtasks in domain_sections.items():
            for subtask_name, schema_info in subtasks.items():
                if subtask_name.replace("_", " ") in prompt_lower:
                    return domain_name, section, subtask_name, schema_info
    return "banking", "general", "unspecified_task", {"inputs": [], "outputs": [], "skills": []}

def _build_extraction_system_prompt(domain, subtask, input_schema, complexity):
    keys = list(input_schema.keys()) if isinstance(input_schema, dict) else input_schema
    steps_count = complexity.get("procedure_length", 5)
    return f"""
    You are a Procedural Memory Task Generator.
    Your goal is to analyze the user prompt and generate a valid JSON object.

    Domain: {domain}
    Subtask: {subtask}

    1. Extract Inputs from: {json.dumps(keys)}
       - If provided in prompt, extract value.
       - Otherwise assign 'unknown'.
       - Do NOT invent new data.

    2. Create approx. {steps_count} steps.

    Response format:
    {{
       "inputs": {{ }},
       "steps": []
    }}
    """

def _llm_generate_and_extract(prompt, system_prompt):
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.45,
        "max_tokens": 2000,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start != -1 else {"inputs": {}, "steps": []}
    except Exception:
        return {"inputs": {}, "steps": []}

def _synthesize_outputs(c_engine, sub_rules, validated_inputs):
    synthesized = {}
    rules = sub_rules.get("outputs", {})
    if "account_id" in rules:
        synthesized["account_id"] = c_engine.generate_account_id(validated_inputs)
    if "transaction_id" in rules:
        synthesized["transaction_id"] = c_engine.generate_transaction_id()
    return c_engine.apply_output_templates(sub_rules, validated_inputs, synthesized)


def generate_from_prompt(
    prompt: str,
    category_key: str,
    category_name: str,
    feedback_policy_name: str = "binary_step",
    perturbation_policy_name: str = "none",
):
    random.seed()

    categories = _load_categories()
    domain_data = _load_domain()
    constraints = _load_constraints()
    complexity = _sample_complexity(category_key, categories)

    c_engine = ConstraintEngine(constraints)
    validator = TaskValidator()

    domain_name, section, subtask, schema_info = _resolve_domain_info(prompt, domain_data)
    sub_rules = c_engine.get_subtask_constraints(domain_name, section, subtask)

    system_prompt = _build_extraction_system_prompt(domain_name, subtask, schema_info.get("inputs", []), complexity)
    llm_result = _llm_generate_and_extract(prompt, system_prompt)

    validated_inputs = c_engine.apply_input_rules(sub_rules, llm_result.get("inputs", {}))
    outputs = _synthesize_outputs(c_engine, sub_rules, validated_inputs)

    try:
        steps = validator.validate_steps({"steps": llm_result.get("steps", [])})["steps"]
    except Exception:
        steps = llm_result.get("steps", [])

    fb = _load_feedback_policies().get("feedback_policies", {}).get(feedback_policy_name, {})
    pt = _load_perturbation_policies().get("perturbation_policies", {}).get(perturbation_policy_name, {})

    feedback_metadata = {
        "name": feedback_policy_name,
        "level": fb.get("level", "step"),
        "style": fb.get("style", "binary"),
        "labels": ["correct", "incorrect"],
        "hint_after_incorrect_n": 4,
    }

    mode_mapping = {
        "none": "off",
        "probabilistic": "probabilistic",
        "progressive": "progressive",
        "performance_based": "adaptive",
    }

    perturbation_metadata = {
        "name": perturbation_policy_name,
        "description": pt.get("description", ""),
        "mode": mode_mapping.get(perturbation_policy_name, "custom"),
    }

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
        "created_at": datetime.now().isoformat(),
        "metadata": {
            "feedback_policy": feedback_metadata,
            "perturbation_policy": perturbation_metadata
        },
    }

        # Save the task file
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATED_DIR / f"{task['task_id']}.json", "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    # Update registry
    registry = _load_json(REGISTRY_PATH) if REGISTRY_PATH.exists() else []
    if not any(r["task_id"] == task["task_id"] for r in registry):
        registry.append({
            "task_id": task["task_id"],
            "category_key": category_key,
            "category_name": category_name,
        })
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    return task

def generate_multiple_instances(
    prompt,
    category_key,
    category_name,
    n_instances=2,
    feedback_policy_name="binary_step",
    perturbation_policy_name="none"
):
    tasks = []
    attempts = 0
    while len(tasks) < n_instances and attempts < n_instances * 3:
        attempts += 1
        try:
            t = generate_from_prompt(
                prompt,
                category_key,
                category_name,
                feedback_policy_name=feedback_policy_name,
                perturbation_policy_name=perturbation_policy_name
            )
            tasks.append(t)
        except Exception as e:
            print(f"Generation failed: {e}")
    return tasks