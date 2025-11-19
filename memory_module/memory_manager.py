import json
import os
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional


def _append_jsonl(path: str, entry: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(path: str, data: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class EpisodicMemory:
    def __init__(self, path="memory/episodic_memory.jsonl"):
        self.path = path

    def log_trial(self, task_id: str, trial_idx: int, input_data: Dict[str, Any],
                  model_plan: List[str], verifier: Dict[str, Any],
                  feedback: List[str], success: bool, lesson_learned: str):
        entry = {
            "task_id": task_id,
            "trial": trial_idx,
            "input": input_data,
            "model_plan": model_plan,
            "verifier": verifier,
            "feedback": feedback,
            "success": success,
            "lesson_learned": lesson_learned,
            "timestamp": datetime.now().isoformat()
        }
        _append_jsonl(self.path, entry)

    def get_recent_trials(self, signature: str, n: int = 4) -> List[Dict[str, Any]]:
        all_trials = _load_jsonl(self.path)
        return [t for t in reversed(all_trials) if t["task_id"].startswith(signature)][:n]



class ProceduralMemory:
    def __init__(self, path="memory/procedural_memory.jsonl"):
        self.path = path
        self.procedures = _load_jsonl(self.path)
        self.index = self._build_index(self.procedures)

    def _build_index(self, procedures):
        index = defaultdict(list)
        for p in procedures:
            index[p["class"]].append(p)
        return index

    # retrieval part have to be arranged, this is just a squelette
    def retrieve(self, signature: str, task_class: Optional[str] = None) -> Optional[Dict[str, Any]]:
        candidates = []
        if task_class and task_class in self.index:
            candidates = self.index[task_class]
        else:
            candidates = [p for group in self.index.values() for p in group]

        # 1. Exact match
        for p in candidates:
            if p["signature"] == signature:
                return p

        # 2. Family-level fallback (symbolic)
        family = signature.split("(")[0]
        family_candidates = [p for p in candidates if p["signature"].startswith(family)]
        if family_candidates:
            return max(family_candidates, key=lambda x: x.get("success_rate", 0))

        # 3. Optional embedding fallback
        if USE_EMBEDDINGS and candidates:
            qvec = _embedder.encode(signature)
            best = max(candidates, key=lambda p: util.cos_sim(qvec, _embedder.encode(p["signature"])))
            return best

        return None

    
    def _save(self):
        _save_jsonl(self.path, self.procedures)

    def _find_proc(self, signature):
        for p in self.procedures:
            if p["signature"] == signature:
                return p
        return None

    def update_after_trials(self, signature: str, task_class: str, recent_trials: List[Dict[str, Any]], plan: List[str]):
        """Implements the stable-improvement rule."""
        k = sum(1 for t in recent_trials if t["success"])
        n = len(recent_trials)
        if n < 3:
            return  # wait for enough evidence

        if k >= 3:  # e.g., 3 successes in last 4
            proc = self._find_proc(signature)
            if not proc:
                # Create new
                new_proc = {
                    "procedure_id": f"{signature}_v{len(self.procedures)+1}",
                    "class": task_class,
                    "signature": signature,
                    "steps": plan,
                    "success_rate": k / n,
                    "last_seen": datetime.now().isoformat()
                }
                self.procedures.append(new_proc)
                print(f"[ProceduralMemory] New procedure stored: {new_proc['procedure_id']}")
            else:
                # Update existing
                proc["success_rate"] = round(0.7 * proc.get("success_rate", 0) + 0.3 * (k / n), 3)
                proc["last_seen"] = datetime.now().isoformat()
                print(f"[ProceduralMemory] Updated procedure: {proc['procedure_id']} (success={proc['success_rate']})")

            self._save()



class MemoryManager:
    def __init__(self):
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()

    def after_trial(self, task_id: str, trial_idx: int, task_class: str,
                    signature: str, input_data: Dict[str, Any], model_plan: List[str],
                    verifier: Dict[str, Any], feedback: List[str], success: bool):
        # feedback summary
        lesson = feedback[-1] if feedback else ("Success" if success else "No feedback")
        self.episodic.log_trial(task_id, trial_idx, input_data, model_plan, verifier, feedback, success, lesson)

        # Retrieve recent trials for same task signature
        recent = self.episodic.get_recent_trials(signature, n=4)
        self.procedural.update_after_trials(signature, task_class, recent, model_plan)
