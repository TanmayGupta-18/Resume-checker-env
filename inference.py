#!/usr/bin/env python3

import os
import json
from typing import Any, Dict, List
from openai import OpenAI
from server.environment import ResumeOptimizationEnv as DeliveryEnv

# ✅ MUST use these (validator requirement)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

VALID_ACTIONS = {0, 1, 2}


def clamp_score(value: float) -> float:
    if value <= 0.0:
        return 0.01
    if value >= 1.0:
        return 0.99
    return value


def fallback_action(state: Dict[str, Any]) -> int:
    agent_pos = state.get("agent_pos", 0)
    pending_orders: List[int] = state.get("pending_orders", [])

    if not pending_orders:
        return 2

    target = pending_orders[0]

    if agent_pos < target:
        return 0
    elif agent_pos > target:
        return 1
    else:
        return 2


def parse_action(text: str, state: Dict[str, Any]) -> int:
    if not text:
        return fallback_action(state)

    cleaned = text.strip()

    try:
        action = int(cleaned)
        if action in VALID_ACTIONS:
            return action
    except Exception:
        pass

    for ch in cleaned:
        if ch in {"0", "1", "2"}:
            return int(ch)

    return fallback_action(state)


# ✅ FIXED: ensures API call is made (retry added)
def choose_action(state: Dict[str, Any]) -> int:
    prompt = f"""
You are controlling a delivery agent in a 1D grid world.

Actions:
0 = move_right
1 = move_left
2 = deliver_order

Current state:
{json.dumps(state)}

Return ONLY one digit: 0, 1, or 2.
""".strip()

    # 🔥 retry once to ensure API call success
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with only one digit: 0, 1, or 2."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=5,
            )

            text = response.choices[0].message.content
            return parse_action(text, state)

        except Exception:
            continue

    return fallback_action(state)


def reward_to_score(total_reward: float, difficulty: str) -> float:
    offset_map = {
        "easy": 20.0,
        "medium": 25.0,
        "hard": 30.0,
    }
    scale_map = {
        "easy": 60.0,
        "medium": 70.0,
        "hard": 80.0,
    }

    offset = offset_map.get(difficulty, 25.0)
    scale = scale_map.get(difficulty, 70.0)

    raw = (total_reward + offset) / scale
    return clamp_score(raw)


def run_episode(task_name: str) -> float:
    env = DeliveryEnv(difficulty=task_name)

    try:
        state = env.reset()
    except Exception:
        print(f"[START] task={task_name}", flush=True)
        print(f"[END] task={task_name} score=0.01 steps=0", flush=True)
        return 0.01

    print(f"[START] task={task_name}", flush=True)

    total_reward = 0.0
    step_count = 0
    done = False
    max_guard_steps = 100

    while not done and step_count < max_guard_steps:
        action = choose_action(state)

        try:
            next_state, reward, done = env.step(action)
        except Exception:
            score = 0.01
            print(
                f"[STEP] task={task_name} step={step_count + 1} action={action} reward=0.00 done=true",
                flush=True,
            )
            print(
                f"[END] task={task_name} score={score:.4f} steps={step_count}",
                flush=True,
            )
            return score

        step_count += 1
        total_reward += reward
        state = next_state

        reward_value = float(reward)
        done_str = "true" if done else "false"

        print(
            f"[STEP] task={task_name} step={step_count} action={action} reward={reward_value:.2f} done={done_str}",
            flush=True,
        )

    score = reward_to_score(total_reward, task_name)

    print(
        f"[END] task={task_name} score={score:.4f} steps={step_count}",
        flush=True,
    )

    return score


def main() -> None:
    for task_name in ["easy", "medium", "hard"]:
        run_episode(task_name)


if __name__ == "__main__":
    main()
