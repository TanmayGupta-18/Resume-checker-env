#!/usr/bin/env python3

import os
import sys
import requests
from typing import List, Dict

# Try OpenAI client (keep your logic intact)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
LITELLM_BASE_URL = "https://litellm.sclr.ac/"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 12

# ---------------- LOGGING (FIXED) ---------------- #

def log_start(task: str):
    print(f"[START] task={task}", flush=True)

def log_step(task: str, step: int, action: str, reward: float, done: bool):
    done_val = "true" if done else "false"
    print(
        f"[STEP] task={task} step={step} action={action} reward={reward:.2f} done={done_val}",
        flush=True,
    )

def log_end(task: str, steps: int, score: float):
    score = max(0.01, min(0.99, score))
    print(
        f"[END] task={task} score={score:.4f} steps={steps}",
        flush=True,
    )

# ---------------- CLIENT ---------------- #

def get_client():
    if not OPENAI_AVAILABLE or not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=LITELLM_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        return None

# ---------------- ACTION LOGIC (UNCHANGED) ---------------- #

def select_action(obs: Dict, client, task: str = "easy"):
    if client:
        try:
            prompt = f"""You are optimizing a resume. Current state: {obs}
Task difficulty: {task}
Choose ONE action from: add_missing_keyword, quantify_achievement, remove_weak_phrase, tailor_summary, reorder_skills, remove_irrelevant_content, strengthen_bullet
Reply with ONLY the action name, nothing else."""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0
            )

            action = response.choices[0].message.content.strip().lower()

            valid_actions = [
                "add_missing_keyword", "quantify_achievement",
                "remove_weak_phrase", "tailor_summary",
                "reorder_skills", "remove_irrelevant_content",
                "strengthen_bullet"
            ]

            if action in valid_actions:
                return action

        except Exception:
            pass

    # fallback (unchanged)
    if task == "hard":
        if obs.get("needs_quantification"):
            return "quantify_achievement"
        if obs.get("needs_weak_cleanup") or obs.get("weak_phrases_count", 0) > 0:
            return "remove_weak_phrase"

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

# ---------------- EPISODE ---------------- #

def run_episode(task: str, client):
    rewards = []
    steps_taken = 0
    final_score = 0.01

    task_max_steps = {"easy": 10, "medium": 15, "hard": 20}
    max_steps = task_max_steps.get(task, MAX_STEPS)

    try:
        r = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task": task, "seed": 42},
            timeout=10
        )
        r.raise_for_status()
        obs = r.json()

        log_start(task)

        done = False

        while not done and steps_taken < max_steps:
            action = select_action(obs, client, task)

            try:
                r = requests.post(
                    f"{API_BASE_URL}/step",
                    json={"name": action},
                    timeout=15
                )
                r.raise_for_status()
                data = r.json()

                obs = data["observation"]
                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))

                rewards.append(reward)
                steps_taken += 1

                final_score = float(obs.get("current_score", final_score))

                log_step(task, steps_taken, action, reward, done)

            except Exception:
                log_step(task, steps_taken + 1, action, 0.0, True)
                break

            if final_score >= 0.95:
                break

    except Exception:
        log_start(task)

    finally:
        try:
            log_end(task, steps_taken, final_score)
        except Exception:
            print(f"[END] task={task} score=0.01 steps=0", flush=True)

# ---------------- MAIN ---------------- #

def main():
    client = get_client()

    for task in TASKS:
        run_episode(task, client)

    sys.exit(0)

if __name__ == "__main__":
    main()
