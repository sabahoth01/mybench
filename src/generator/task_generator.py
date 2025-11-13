import os
import json
import random
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

GENERATED_DIR = Path("data/generated_tasks")
REGISTRY_PATH = Path("data/task_registry.json")
DOMAIN_FILE = Path("configs/task_domaine.json")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
# qwen
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_domain():
    with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
## test qwen prompt following capacity// must be corrected according to my methodology file
def _llm_generate_task(prompt, category_key, category_name, domain_name="banking"):
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
        "max_tokens": 500,
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

    if "task_id" not in task:
        task["task_id"] = f"{category_key}_{int(datetime.now().timestamp())}_{random.randint(100,999)}"

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