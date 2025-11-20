import os
import json
import random
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import uuid

GENERATED_DIR = Path("data/generated_tasks")
REGISTRY_PATH = Path("data/task_registry.json")
CATEGORY_FILE = Path("configs/test_category.json")
DOMAIN_FILE = Path("configs/task_domaine.json")
KEYWORD_FILE = Path("configs/domain_keywords.json")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
# qwen
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _load_domain():
    if not DOMAIN_FILE.exists():
        raise FileNotFoundError(f"Domain file {DOMAIN_FILE} not found.")
    with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_keywords():
    if not KEYWORD_FILE.exists():
        raise FileNotFoundError(f"Keyword file {KEYWORD_FILE} not found.")
    with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_categories():
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _generate_task_id(category_key: str) -> str:
    return f"{category_key}_{uuid.uuid4().hex[:6]}"
 
def _sample_complexity(category_key: str, categories: dict):
    """
    Randomly samples complexity parameters for a given category.
    """
    if category_key not in categories:
        raise ValueError(f"Unknown category {category_key}")

    complexity_meta = categories[category_key].get("complexity", {})
    if not complexity_meta:
        return {}

    sampled = {"category": category_key}

    for key, value in complexity_meta.items():
        # If the value is a list of integers, pick one randomly
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            sampled[key] = random.choice(value)
        # If the value is already a single int or str, just use it
        else:
            sampled[key] = value
    return sampled


def _resolve_domain_info(prompt: str, domain_data: dict):
    """
    Fuzzy domain/subtask detection using external keyword file
    Returns best matching (domain, section, subtask, schema_info),
    or fallback.
    """
    prompt_lower = prompt.lower()
    keyword_map = _load_keywords()

    # First pass: keyword-based matching
    for domain_name, domain_sections in domain_data.items():
        for section, subtasks in domain_sections.items():
            for subtask_name, schema_info in subtasks.items():
                # keywords loaded from external file
                keywords = keyword_map.get(subtask_name, [])
                for kw in keywords:
                    if kw in prompt_lower:
                        return domain_name, section, subtask_name, schema_info
                # secondary: partial match on subtask name
                words = subtask_name.replace("_", " ").split()
                if all(w in prompt_lower for w in words):
                    return domain_name, section, subtask_name, schema_info
    # fallback
    return "banking", "general", "unspecified_task", {
        "inputs": [],
        "outputs": [],
        "skills": []
    }

def _llm_generate_task(prompt, category_key, category_name, domain_data, complexity=None):
    """
    Calls OpenRouter LLM to produce a domain-grounded task instance.
    """
    domain_name, section, subtask, schema_info = _resolve_domain_info(prompt, domain_data)
    complexity_info = f"- Complexity: {complexity}" if complexity else ""
    system_prompt = f"""
    You are a procedural task generator for a benchmark.
    Given a user prompt and a domain schema, generate a concrete, realistic task instance.
    The output must be **pure JSON** only (no text outside JSON).

    ### Context
    - Domain: {domain_name}
    - Section: {section}
    - Subtask: {subtask}
    - Complexity: {complexity_info}

    ### Schema
    - Inputs: {schema_info['inputs']}
    - Outputs: {schema_info['outputs']}
    - Required Skills: {schema_info['skills']}

    ### Rules
    - Task must follow the schema logically.
    - Include steps that use the listed inputs and produce the listed outputs.
    - Keep tasks coherent with the benchmark category ({category_name} / {category_key}).
    - Generate exactly one JSON with fields:
    task_id, category_key, category_name, domain, section, subtask, steps, inputs, outputs, skills.

    ### Complexity Profile
    The task must match this complexity specification:
    {json.dumps(complexity, indent=2)}

    Respect:
    - Procedure length target
    """
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Prompt: {prompt}"}
        ],
        "temperature": 0.6,
        "max_tokens": 2000,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    raw_output = data["choices"][0]["message"]["content"]
    try:
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        return json.loads(raw_output[json_start:json_end])
    except Exception as e:
        raise ValueError(f"Model output not valid JSON: {raw_output}") from e

def generate_from_prompt(prompt: str, category_key: str, category_name: str):
    """
    Hybrid LLM + schema-guided generator.
    """
    random.seed()
    categories = _load_categories()
    domain_data = _load_domain()
    complexity = _sample_complexity(category_key, categories)
    task = _llm_generate_task(prompt, category_key, category_name, domain_data, complexity)
    task["task_id"] = _generate_task_id(category_key)

    task.setdefault("category_key", category_key)
    task.setdefault("category_name", category_name)
    task.setdefault("domain", domain_data)
    task["complexity"] = complexity
    task["prompt"] = prompt
    task["created_at"] = datetime.now().isoformat()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATED_DIR / f"{task['task_id']}.json", "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    if REGISTRY_PATH.exists():
        registry = json.load(open(REGISTRY_PATH, encoding="utf-8"))
    else:
        registry = []

    registry.append({
        "task_id": task["task_id"],
        "category_key": category_key,
        "category_name": category_name,
    })
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    return task

def generate_multiple_instances(prompt, category_key, category_name, n_instances: int = 2):
    tasks = []
    attempts = 0
    max_attempts = n_instances * 3  
    while len(tasks) < n_instances and attempts < max_attempts:
        attempts += 1
        try:
            t = generate_from_prompt(prompt, category_key, category_name)
            
            if t["task_id"] not in [task["task_id"] for task in tasks]:
                tasks.append(t)
        except Exception as e:
            print(f"LLM generation failed: {e}. Retrying... ({len(tasks)+1}/{n_instances})")
    if len(tasks) < n_instances:
        print(f"Only generated {len(tasks)} tasks out of requested {n_instances}")
    return tasks