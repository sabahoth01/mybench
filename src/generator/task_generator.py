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
DOMAIN_FILE = Path("configs/task_domaine.json")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
# qwen
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_domain():
    if not DOMAIN_FILE.exists():
        raise FileNotFoundError(f"Domain file {DOMAIN_FILE} not found.")
    with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
## test qwen prompt following capacity// must be corrected according to my methodology file

def _generate_task_id(category_key: str) -> str:
    return f"{category_key}_{uuid.uuid4().hex[:6]}"

def _llm_generate_task(prompt, category_key, category_name, domain_name="banking sphere"):
    """
    Calls OpenRouter LLM to produce a dynamic task instance.
    """
    system_prompt = f"""
You are a task generator for a LLM procedural memory benchmark.
Given a domain schema and a user prompt, you will generate a *concrete, realistic task instance*.
Output must be valid JSON only (no text outside JSON).

### Instructions:
- Domain: {domain_name}
- Category: {category_name} ({category_key})
- Rule for B1.1 ("simple atomic recall"): produce a short, self-contained, single-task procedure (2–10 steps max).
- Each step should be clearly written as an instruction.
- Include realistic parameter values consistent with the schema.
- Use JSON fields: task_id, category_key, category_name, domain, steps, variables.
"""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Prompt: {prompt}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    raw_output = data["choices"][0]["message"]["content"]
    # try to extract JSON
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
    domain = _load_domain()
    domain_name = "banking sphere"  # for now; later can be user-selectable from each (sub)domaine that i will finalised

    task = _llm_generate_task(prompt, category_key, category_name, domain_name)

    task["task_id"] = _generate_task_id(category_key)

    task.setdefault("category_key", category_key)
    task.setdefault("category_name", category_name)
    task.setdefault("domain", domain_name)
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
            # ensure task_id is unique
            if t["task_id"] not in [task["task_id"] for task in tasks]:
                tasks.append(t)
        except Exception as e:
            print(f"LLM generation failed: {e}. Retrying... ({len(tasks)+1}/{n_instances})")
    if len(tasks) < n_instances:
        print(f"Only generated {len(tasks)} tasks out of requested {n_instances}")
    return tasks
