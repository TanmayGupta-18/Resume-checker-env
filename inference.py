#!/usr/bin/env python3
"""
Strict inference.py for OpenEnv Phase 2
LiteLLM proxy + safe client creation
"""

import os
import sys
import requests
from typing import List, Optional, Dict

# Try to import OpenAI safely
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
LITELLM_BASE_URL = "https://litellm.sclr.ac/"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_NAME = os.getenv("ENV_NAME", "resume-optimization")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 12

def get_client():
    if not OPENAI_AVAILABLE or not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=LITELLM_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        return None

def log_start(task: str):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def select_action(obs: Dict, client, task: str = "easy"):
    if task == "hard":
        if obs.get("needs_quantification"):
            return "quantify_achievement"
        if obs.get("needs_weak_cleanup") or obs.get("weak_phrases_count", 0) > 0:
            return "remove_weak_phrase"
    # General heuristic
    if obs.get("needs_keyword_work") and obs.get("missing_keywords"):
        return "add_missing_keyword"
    if obs.get("needs_quantification"):
        return "quantify_achievement"
    if obs.get("needs_weak_cleanup") or obs.get("weak_phrases_count", 0) > 0:
        return "remove_weak_phrase"
    if not obs.get("summary_tailored", True):
        return "tailor_summary"
    if obs.get("needs_skill_reorder"):
        return "reorder_skills"
    if obs.get("irrelevant_skills_count", 0) > 0:
        return "remove_irrelevant_content"
    return "strengthen_bullet"

def run_episode(task: str, client):
    rewards = []
    steps_taken = 0
    final_score = 0.0
    success = False

    # Per-task step limits
    task_max_steps = {"easy": 10, "medium": 15, "hard": 20}
    max_steps = task_max_steps.get(task, MAX_STEPS)

    try:
        r = requests.post(f"{API_BASE_URL}/reset", json={"task": task, "seed": 42}, timeout=10)
        r.raise_for_status()
        obs = r.json()

        log_start(task)

        done = False
        while not done and steps_taken < max_steps:
            action = select_action(obs, client, task)

            try:
                r = requests.post(f"{API_BASE_URL}/step", json={"name": action}, timeout=15)
                r.raise_for_status()
                data = r.json()

                obs = data["observation"]
                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))
                step_error = data.get("error", None)

                rewards.append(reward)
                steps_taken += 1
                final_score = max(0.0, min(1.0, float(obs.get("current_score", 0.0))))

                log_step(steps_taken, action, reward, done, step_error)

            except requests.RequestException as e:
                log_step(steps_taken + 1, action, 0.0, False, str(e))
                break

            if final_score >= 0.95:
                break

        success = final_score >= 0.55

    except Exception as e:
        print(f"[ERROR] {task}: {e}", flush=True)

    finally:
        # Always emit [END] even on exception
        log_end(success, steps_taken, final_score, rewards)

def main():
    client = get_client()

    for task in TASKS:
        run_episode(task, client)

    sys.exit(0)

if __name__ == "__main__":
    main()
