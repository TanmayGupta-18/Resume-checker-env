import os
import requests
import time
from openai import OpenAI

# ── CONFIG ─────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MAX_TOKENS = 200

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 15

# ── OPENAI CLIENT (HF ROUTER) ─────────────────────────────

api_key = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

# ── PROMPT ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert resume optimizer.

Return ONLY one action from:
add_missing_keyword, quantify_achievement, remove_weak_phrase,
tailor_summary, reorder_skills, remove_irrelevant_content, strengthen_bullet
"""

VALID_ACTIONS = [
    "add_missing_keyword",
    "quantify_achievement",
    "remove_weak_phrase",
    "tailor_summary",
    "reorder_skills",
    "remove_irrelevant_content",
    "strengthen_bullet"
]

def build_prompt(obs):
    return f"""
Score: {obs['current_score']}
Missing keywords: {obs['missing_keywords']}
Weak phrases: {obs['weak_phrases_count']}
Quantified: {obs['quantified_bullets']}/{obs['total_bullets']}

What is the best next action?
"""

# ── ACTION SELECTION ─────────────────────────────────────

def select_action(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs)}
            ]
        )

        action = response.choices[0].message.content.strip().lower()

        if action in VALID_ACTIONS:
            return action

    except Exception:
        pass

    # 🔒 Fallback 
    if obs["needs_keyword_work"] and obs["missing_keywords"]:
        return "add_missing_keyword"

    if obs["quantified_bullets"] < obs["total_bullets"]:
        return "quantify_achievement"

    if obs["weak_phrases_count"] > 0:
        return "remove_weak_phrase"

    if not obs["summary_tailored"]:
        return "tailor_summary"

    if obs["needs_skill_reorder"]:
        return "reorder_skills"

    if obs["irrelevant_skills_count"] > 0:
        return "remove_irrelevant_content"

    return "strengthen_bullet"

# ── EPISODE RUNNER ──────────────────────────────────────

def run_episode(task):
    # RESET
    res = requests.post(f"{API_BASE_URL}/reset", json={"task": task})
    res.raise_for_status()
    obs = res.json()

    print(f"[START] task={task}")

    steps = 0
    done = False

    while not done and steps < MAX_STEPS:
        action = select_action(obs)

        step_res = requests.post(
            f"{API_BASE_URL}/step",
            json={"name": action}
        )
        step_res.raise_for_status()

        data = step_res.json()
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]

        print(
            f"[STEP] step={steps+1} "
            f"action={action} "
            f"reward={round(reward,4)} "
            f"score={round(obs['current_score'],4)}"
        )

        steps += 1

        if obs["current_score"] >= 0.85:
            break

    grade_res = requests.get(f"{API_BASE_URL}/grade")
    grade_res.raise_for_status()
    grade = grade_res.json()

    print(
        f"[END] task={task} "
        f"final_score={round(grade['final_score'],4)} "
        f"grade={round(grade['grade'],4)}"
    )

    return grade


# ── MAIN ────────────────────────────────────────────────

def main():
    start = time.time()
    results = []

    for task in TASKS:
        try:
            results.append(run_episode(task))
        except Exception as e:
            print(f"[ERROR] task={task} error={str(e)}")

    avg = sum(r["final_score"] for r in results) / max(1, len(results))

    print(f"\n[SUMMARY] avg_score={round(avg,4)} tasks={len(results)}")
    print(f"[TIME] total_seconds={round(time.time()-start,2)}")


if __name__ == "__main__":
    main()