# server/app.py

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from models import (
    Action,
    ResetRequest,
    StepResponse,
    GradeResponse,
    Observation,
    VALID_ACTIONS
)

from server.environment import ResumeOptimizationEnv
from pydantic import BaseModel

class ResetRequest(BaseModel):
    task: str = "default"
    seed: int = 0


app = FastAPI(
    title="Resume Optimization Environment",
    version="2.0.0",
    description="OpenEnv - Resume optimization RL environment"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 🔧 Helper Functions
# -------------------------

def _get_env(req: Request):
    env = getattr(req.app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first")
    return env


def compute_resume_diff(old, new):
    diff = {}

    for key in old:
        if key in new and old[key] != new[key]:
            diff[key] = {
                "before": old[key],
                "after": new[key]
            }

    return diff


def _state_to_observation(state: dict):
    return Observation(
        current_score=state["current_score"],
        steps_taken=state["steps_taken"],
        steps_remaining=state["steps_remaining"],
        missing_keywords=state["missing_keywords"],
        weak_phrases_count=state["weak_phrases_count"],
        quantified_bullets=state["quantified_bullets"],
        total_bullets=state["total_bullets"],
        skill_match_ratio=state["skill_match_ratio"],
        summary_tailored=state["summary_tailored"],
        irrelevant_skills_count=state["irrelevant_skills_count"],
        needs_keyword_work=state["needs_keyword_work"],
        needs_quantification=state["needs_quantification"],
        needs_weak_cleanup=state["needs_weak_cleanup"],
        needs_summary_work=state["needs_summary_work"],
        needs_skill_reorder=state["needs_skill_reorder"],
        resume_comparison=state.get("resume_comparison", {"summary": "Initial state"}),
        resume=state["resume"],
        job_description=state["job_description"],
    )

# -------------------------
# 🚀 API Endpoints
# -------------------------

@app.get("/")
def root():
    return {"name": "Resume Optimization Environment", "actions": VALID_ACTIONS}


# 🔁 RESET
@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest, req: Request):
    env = ResumeOptimizationEnv(task=request.task, seed=request.seed)
    req.app.state.env = env

    state = env.reset()
    return _state_to_observation(state)


# ⚡ STEP
@app.post("/step", response_model=StepResponse)
def step(action: Action, req: Request):
    env = _get_env(req)

    if action.name not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{action.name}'. Valid actions: {VALID_ACTIONS}"
        )

    new_state, reward, done, info = env.step(action.name)

    return StepResponse(
        observation=_state_to_observation(new_state),
        reward=reward,
        done=done,
        info=info
    )


# 🧠 SMART AGENT
@app.post("/run_episode", response_model=GradeResponse)
def run_episode(request: ResetRequest, req: Request):
    env = ResumeOptimizationEnv(task=request.task, seed=request.seed)
    req.app.state.env = env

    state = env.reset()
    initial_state = state

    steps_taken = 0
    max_steps = 15

    last_action = None
    last_reward = None

    action_rewards = {a: [] for a in VALID_ACTIONS}

    print(f"\n🚀 Running episode for: {request.task.upper()}")

    while steps_taken < max_steps and not env._done:
        obs = env.state()

        # 🚫 Avoid bad actions
        bad_actions = {
            a for a, rewards in action_rewards.items()
            if len(rewards) >= 2 and sum(rewards[-2:]) < 0
        }

        # 🎯 Candidate actions
        candidates = []

        if obs["needs_keyword_work"] and obs["missing_keywords"]:
            candidates.append("add_missing_keyword")

        if obs["needs_quantification"]:
            candidates.append("quantify_achievement")

        if obs["needs_weak_cleanup"]:
            candidates.append("remove_weak_phrase")

        if not obs["summary_tailored"]:
            candidates.append("tailor_summary")

        if obs["needs_skill_reorder"]:
            candidates.append("reorder_skills")

        if obs["irrelevant_skills_count"] > 0:
            candidates.append("remove_irrelevant_content")

        candidates.append("strengthen_bullet")

        # ❌ Remove bad actions
        candidates = [a for a in candidates if a not in bad_actions]

        # 🔁 Avoid repetition
        if last_action in candidates and len(candidates) > 1:
            candidates.remove(last_action)

        # 🎯 Choose action
        action = random.choice(candidates)

        try:
            new_state, reward, done, info = env.step(action)
            steps_taken += 1

            action_rewards[action].append(reward)
            last_action = action
            last_reward = reward

            print(f"  Step {steps_taken:2d} | {action:<25} | "
                  f"Score: {new_state['current_score']:.3f} | "
                  f"Reward: {reward:.3f}")

        except Exception as e:
            print(f"  ⚠️ Skipped {action}: {str(e)}")
            break

        

    # ✅ AFTER LOOP (correct place)
    final_state = env.state()
    final_grade = env.grade()

    print(f"\n✅ Final Score: {final_state['current_score']:.3f} | Grade: {final_grade:.3f}\n")

    return GradeResponse(
        grade=final_grade,
        task=request.task,
        steps_taken=steps_taken,
        final_score=final_state["current_score"],
        resume=final_state["resume"],
        diff=compute_resume_diff(initial_state["resume"], final_state["resume"])
    )


# 📊 STATE
@app.get("/state", response_model=Observation)
def get_state(req: Request):
    env = _get_env(req)
    return _state_to_observation(env.state())


# 🏁 GRADE
@app.get("/grade", response_model=GradeResponse)
def grade(req: Request):
    env = _get_env(req)
    current = env.state()

    return GradeResponse(
        grade=env.grade(),
        task=env.task,
        steps_taken=current.get("steps_taken", 0),
        final_score=current.get("current_score", 0.0),
        resume=current.get("resume", ""),
        diff={}
    )


# 📜 ACTIONS
@app.get("/actions")
def list_actions():
    return {
        "actions": VALID_ACTIONS,
        "descriptions": {
            "add_missing_keyword": "Add required keyword",
            "quantify_achievement": "Add measurable metrics",
            "remove_weak_phrase": "Replace weak phrases",
            "reorder_skills": "Reorder skills",
            "tailor_summary": "Customize summary",
            "add_relevant_project": "Highlight relevant project",
            "remove_irrelevant_content": "Remove irrelevant info",
            "strengthen_bullet": "Improve bullet wording",
        }
    }


# ▶ RUN SERVER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)
