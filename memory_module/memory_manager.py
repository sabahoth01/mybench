import json
import os
import hashlib
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# -------------------------
# JSONL helpers (unchanged)
# -------------------------
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

# ---------------------------------------
# Small deterministic "embedding" utility
# (no external libs; deterministic for testing)
# ---------------------------------------
def generate_embedding(steps: List[str], dim: int = 64) -> List[float]:
    """
    Produce a deterministic embedding vector for a list of plan steps.
    This is a lightweight, deterministic substitute for a learned embedding model.
    It uses SHA256 hashing of step strings and folds bytes into a float vector,
    then L2-normalizes.
    """
    if not steps:
        return [0.0] * dim
    joined = "\n".join(steps)
    h = hashlib.sha256(joined.encode("utf-8")).digest()  # 32 bytes
    # Expand to dim floats by repeated hashing
    vec = [0.0] * dim
    base = h
    idx = 0
    while idx < dim:
        # hash previous digest to get more bytes
        base = hashlib.sha256(base).digest()
        for b in base:
            if idx >= dim:
                break
            # map byte -> [-1,1] float
            vec[idx] = (b / 255.0) * 2.0 - 1.0
            idx += 1
    # L2-normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)

# -------------------------
# Kalman filter utilities
# -------------------------
def kalman_update(mean: float, P: float, measurement: float, R: float) -> Tuple[float, float]:
    """
    Simple scalar Kalman update for a single observation.
    mean: prior mean estimate
    P: prior uncertainty (variance)
    measurement: observed value (0 or 1 for success; could be fractional if aggregated)
    R: measurement variance (how noisy a single observation is)
    Returns: (posterior_mean, posterior_P)
    Equations:
       K = P / (P + R)
       mean = mean + K * (measurement - mean)
       P = (1 - K) * P
    """
    K = P / (P + R) if (P + R) > 0 else 0.0
    new_mean = mean + K * (measurement - mean)
    new_P = (1 - K) * P
    # numerical safeguards
    new_mean = max(0.0, min(1.0, new_mean))
    new_P = max(1e-6, new_P)
    return new_mean, new_P

# -------------------------
# Memory classes (reworked)
# -------------------------
class EpisodicMemory:
    """
    Append-only trial log. Uses 'test_category' instead of 'signature'.
    Stores trials for auditing and for generating recent-trial windows.
    """
    def __init__(self, path: str = "memory_module/memory/episodic_memory.jsonl"):
        self.path = path

    def log_trial(self, task_id: str, trial_idx: int, test_category: str,
                  input_data: Dict[str, Any], model_plan: List[str],
                  verifier: Dict[str, Any], feedback: List[str], success: bool):
        entry = {
            "task_id": task_id,
            "trial": trial_idx,
            "test_category": test_category,
            "input": input_data,
            "model_plan": model_plan,
            "verifier": verifier,
            "feedback": feedback,
            "success": bool(success),
            "timestamp": datetime.now().isoformat()
        }
        _append_jsonl(self.path, entry)

    def get_recent_trials(self, test_category: str, n: int = 4) -> List[Dict[str, Any]]:
        all_trials = _load_jsonl(self.path)
        matching = [t for t in all_trials if t.get("test_category") == test_category]
        return matching[-n:] if n > 0 else matching


class ProceduralMemory:
    """
    Stores learned procedures (recipes) along with:
     - embedding vector for the plan
     - Kalman-filtered success estimate and uncertainty P
     - sample_count
     - last_seen timestamp
     - scoring metadata
    """
    def __init__(self, path: str = "memory_module/memory/procedural_memory.jsonl"):
        self.path = path
        self.procedures: List[Dict[str, Any]] = _load_jsonl(self.path)

    def _persist(self):
        _save_jsonl(self.path, self.procedures)

    # ---- retrieval ----
    def retrieve(self, test_category: str, model_plan: Optional[List[str]] = None,
                 top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve best matching procedures for this test_category.
        Steps:
         1. Exact-match on test_category (return top_k by score).
         2. Family-level fallback (prefix before '_'):
             - use cosine similarity between requested model_plan (if provided) and stored embeddings,
             - compute a composite score and return top_k winners.
        The returned list is sorted by descending score.
        """
        # Exact matches
        exact = [p for p in self.procedures if p["test_category"] == test_category]
        if exact:
            scored = sorted(exact, key=lambda x: x.get("score", 0.0), reverse=True)
            return scored[:top_k]

        # Family fallback
        family = test_category.split("_")[0] if "_" in test_category else test_category
        family_candidates = [p for p in self.procedures if p["test_category"].startswith(family)]

        if not family_candidates:
            return []

        # If we have a model_plan provided, compute its embedding for similarity
        query_emb = generate_embedding(model_plan) if model_plan else None

        # compute composite score for each candidate
        scored_candidates = []
        for p in family_candidates:
            success_mean = p.get("success_mean", 0.5)
            P = p.get("P", 1.0)
            # stability = inverse uncertainty
            stability = 1.0 / (1.0 + P)
            # recency factor: more recent => closer to 1
            last_seen = datetime.fromisoformat(p.get("last_seen"))
            days_since = (datetime.now() - last_seen).days
            recency = 1.0 / (1.0 + days_since / 30.0)  # 30-day half-ish scale

            sim = 0.0
            if query_emb is not None and p.get("embedding"):
                sim = cosine_similarity(query_emb, p["embedding"])
                # map sim from [-1,1] to [0,1]
                sim = (sim + 1.0) / 2.0

            # final composite score - weights can be tuned
            score = (0.45 * success_mean) + (0.25 * stability) + (0.15 * recency) + (0.15 * sim)
            # incorporate sample_count as a tiny bonus
            score *= (1.0 + math.log1p(p.get("sample_count", 0)) * 0.02)

            p_copy = dict(p)
            p_copy["score"] = round(score, 4)
            p_copy["similarity"] = round(sim, 4)
            p_copy["recency"] = round(recency, 4)
            p_copy["stability"] = round(stability, 4)
            scored_candidates.append(p_copy)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    def _find_proc_index(self, test_category: str) -> int:
        for i, p in enumerate(self.procedures):
            if p["test_category"] == test_category:
                return i
        return -1

    # ---- update using Kalman ----
    def update_procedure(self, test_category: str, recent_trials: List[Dict[str, Any]]):
        """
        Update or create a procedure using Kalman-style updates.
        recent_trials should be ordered chronologically; we will process them in order.
        Each trial has 'success' boolean and 'model_plan' (the steps).
        Learning rule:
         - If we have >= 3 trials and empirical success proportion suggests improvement,
           we will either create a procedure or update the existing one using Kalman updates.
         - We still require some minimal evidence: at least 3 trials.
        """
        n = len(recent_trials)
        if n < 3:
            return  # insufficient evidence

        # compute empirical success proportion
        successes = [1.0 if t["success"] else 0.0 for t in recent_trials]
        k = int(sum(successes))
        empirical_rate = sum(successes) / n

        # threshold for considering a stable improvement (configurable)
        # require at least 3 successes out of 4 or empirical_rate >= 0.75 for variable n
        if not (k >= 3 or empirical_rate >= 0.75):
            return

        idx = self._find_proc_index(test_category)

        # choose best_plan = model_plan of the last successful trial (if any), else last trial
        success_trials = [t for t in recent_trials if t["success"]]
        if success_trials:
            best_plan = success_trials[-1]["model_plan"]
        else:
            best_plan = recent_trials[-1]["model_plan"]

        emb = generate_embedding(best_plan)

        # If creating: initialize Kalman prior to a conservative estimate
        if idx == -1:
            # initial mean ~ empirical_rate, but keep uncertainty high (P large) so updates adapt
            init_mean = empirical_rate
            init_P = 0.5  # large uncertainty
            # perform sequential Kalman updates through the recent observations to get posterior
            mean, P = init_mean, init_P
            # use per-measurement variance R; we can choose R proportional to Bernoulli variance
            for z in successes:
                # choose R: smaller when many samples, larger for single Bernoulli measurements
                R = max(0.01, z * (1 - z) if (z in (0.0, 1.0)) else 0.05)
                mean, P = kalman_update(mean, P, z, R)
            new_proc = {
                "procedure_id": f"{test_category}_v{len(self.procedures)+1}",
                "test_category": test_category,
                "steps": best_plan,
                "embedding": emb,
                "success_mean": round(mean, 4),
                "P": round(P, 6),
                "sample_count": n,
                "last_seen": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            # compute initial score (for convenience)
            new_proc["score"] = round(0.45 * new_proc["success_mean"] + 0.25 * (1.0 / (1.0 + new_proc["P"])), 4)
            self.procedures.append(new_proc)
            print(f"   [Memory] New procedure learned: {new_proc['procedure_id']} (k={k}/{n}, mean={new_proc['success_mean']})")
        else:
            # update existing via Kalman sequential updates using the recent trials as new observations
            proc = self.procedures[idx]
            mean = proc.get("success_mean", 0.5)
            P = proc.get("P", 0.5)
            # Increase sample count
            proc["sample_count"] = proc.get("sample_count", 0) + n
            for z in successes:
                R = max(0.01, z * (1 - z) if (z in (0.0, 1.0)) else 0.05)
                mean, P = kalman_update(mean, P, z, R)
            proc["success_mean"] = round(mean, 4)
            proc["P"] = round(P, 6)
            proc["last_seen"] = datetime.now().isoformat()
            proc["embedding"] = emb  # optionally refresh embedding if plan changed
            # recompute score
            stability = 1.0 / (1.0 + proc["P"])
            proc["score"] = round((0.45 * proc["success_mean"]) + (0.25 * stability), 4)
            print(f"   [Memory] Procedure updated: {proc['procedure_id']} (new_mean={proc['success_mean']}, P={proc['P']})")

        # persist to disk
        self._persist()

# -------------------------
# Orchestration manager
# -------------------------
class MemoryManager:
    def __init__(self):
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()

    def after_trial(self, task_id: str, trial_idx: int, test_category: str,
                    input_data: Dict[str, Any], model_plan: List[str],
                    verifier: Dict[str, Any], feedback: List[str], success: bool):
        """
        Logs the trial and then attempts to update procedural memory using
        the most recent trials for this test_category.
        """
        # 1. Log episodic
        self.episodic.log_trial(
            task_id=task_id,
            trial_idx=trial_idx,
            test_category=test_category,
            input_data=input_data,
            model_plan=model_plan,
            verifier=verifier,
            feedback=feedback,
            success=success
        )
        # 2. Look at last N trials (N configurable)
        recent = self.episodic.get_recent_trials(test_category, n=4)
        self.procedural.update_procedure(test_category, recent)

    def retrieve_procedure(self, test_category: str, candidate_plan: Optional[List[str]] = None, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Ask procedural memory for best matching procedures before attempting a new trial.
        If candidate_plan is provided, similarity will be used in family fallback scoring.
        Returns up to top_k procedures (sorted by score).
        """
        return self.procedural.retrieve(test_category, model_plan=candidate_plan, top_k=top_k)