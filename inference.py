#!/usr/bin/env python3
"""
OpenEnv Inference Script - Resume Optimization Environment
Using Scaler's LiteLLM Proxy: https://litellm.sclr.ac/
"""

import os
import sys
import requests
from typing import List, Optional, Dict
from openai import OpenAI

# ── CONFIG ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
LITELLM_BASE_URL = "https://litellm.sclr.ac/"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.60

# ── CLIENT SETUP ───────────────────────────────────────────
def get_client():
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not found → using heuristic only", flush=True)
        return None
    try:
        return OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        print(f"[WARN] Failed to create LiteLLM client: {e}", flush=True)
        return None


client = None


# ── MANDATORY LOGGING (Exact Format Required) ──────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = str(error) if error is not None else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── ACTION SELECTION (LLM via LiteLLM + Heuristic fallback) ─
def select_action(obs: Dict, llm_client) -> str:
    VALID_ACTIONS = [
        "add_missing_keyword", "quantify_achievement", "remove_weak_phrase",
        "tailor_summary", "reorder_skills", "remove_irrelevant_content", "strengthen_bullet"
    ]

    if llm_client:
        try:
            prompt = f"""
Current Score: {obs.get('current_score', 0):.3f}
Missing keywords: {len(obs.get('missing_keywords', []))}
Weak phrases: {obs.get('weak_phrases_count', 0)}
Quantified bullets: {obs.get('quantified_bullets', 0)}/{obs.get('total_bullets', 0)}
Summary tailored: {obs.get('summary_tailored', False)}
Irrelevant skills: {obs.get('irrelevant_skills_count', 0)}

Choose **exactly one** action from this list only:
add_missing_keyword, quantify_achievement, remove_weak_phrase, tailor_summary, 
reorder_skills, remove_irrelevant_content, strengthen_bullet

Reply with only the action name.
"""

            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert resume optimizer. Always reply with only the exact action name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=25,
            )

            action = response.choices[0].message.content.strip().lower()
            if action in VALID_ACTIONS:
                return action
        except Exception as e:
            print(f"[DEBUG] LiteLLM failed: {e}", flush=True)

    # Reliable heuristic fallback
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


# ── RUN EPISODE ────────────────────────────────────────────
def run_episode(task: str, llm_client):
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0

    try:
        # Reset
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset", 
            json={"task": task, "seed": 42}, 
            timeout=10
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        log_start(task=task, env="resume-optimization", model=MODEL_NAME)

        done = False
        while not done and steps_taken < MAX_STEPS:
            action = select_action(obs, llm_client)

            step_resp = requests.post(
                f"{API_BASE_URL}/step", 
                json={"name": action}, 
                timeout=15
            )
            step_resp.raise_for_status()
            data = step_resp.json()

            obs = data["observation"]
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))

            rewards.append(reward)
            steps_taken += 1
            final_score = float(obs.get("current_score", 0.0))

            log_step(step=steps_taken, action=action, reward=reward, done=done)

            if final_score >= 0.85:
                break

        success = final_score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

        return {"success": success, "steps": steps_taken, "score": final_score, "rewards": rewards}

    except Exception as e:
        print(f"[ERROR] task={task}: {e}", flush=True)
        log_end(success=False, steps=steps_taken, score=final_score, rewards=rewards)
        return {"success": False, "steps": steps_taken, "score": final_score, "rewards": rewards}


# ── MAIN ───────────────────────────────────────────────────
def main():
    global client
    print("=== Resume Optimization - LiteLLM Inference Started ===", flush=True)

    client = get_client()

    # Server health check
    try:
        resp = requests.get(f"{API_BASE_URL}/", timeout=5)
        if resp.status_code != 200:
            print("❌ Server not reachable at", API_BASE_URL, flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Server connection error: {e}", flush=True)
        sys.exit(1)

    all_results = []
    for task in TASKS:
        result = run_episode(task, client)
        all_results.append(result)

    avg_score = sum(r.get("score", 0) for r in all_results) / len(all_results)
    print(f"\n[SUMMARY] Average Score: {avg_score:.3f} | Tasks: {len(TASKS)}", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
