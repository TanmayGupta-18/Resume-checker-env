# models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional

class Action(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        ..., 
        description="One of the 8 resume optimization actions",
        examples=["tailor_summary"]   # ← This helps Swagger
    )


class Observation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Progress signals
    current_score: float = Field(..., ge=0.0, le=1.0)
    steps_taken: int = Field(...)
    steps_remaining: int = Field(...)

    # Specific improvement signals
    missing_keywords: List[str] = Field(default_factory=list)
    weak_phrases_count: int = Field(default=0)
    quantified_bullets: int = Field(default=0)
    total_bullets: int = Field(default=0)
    skill_match_ratio: float = Field(default=0.0)
    summary_tailored: bool = Field(default=False)
    irrelevant_skills_count: int = Field(default=0)

    # Boolean hints
    needs_keyword_work: bool = Field(default=False)
    needs_quantification: bool = Field(default=False)
    needs_weak_cleanup: bool = Field(default=False)
    needs_summary_work: bool = Field(default=False)
    needs_skill_reorder: bool = Field(default=False)

    # Resume comparison - FIXED TYPING
    resume_comparison: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shows what changed from the previous resume state. Empty on first reset."
    )

    # Raw context - FIXED TYPING
    resume: Dict[str, Any] = Field(...)
    job_description: Dict[str, Any] = Field(...)

class StepResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task: str = Field(default="easy")
    seed: Optional[int] = Field(default=None)

class GradeResponse(BaseModel):
    grade: float
    task: str
    steps_taken: int
    final_score: float
    

VALID_ACTIONS = [
    "add_missing_keyword", "quantify_achievement", "remove_weak_phrase",
    "reorder_skills", "tailor_summary", "add_relevant_project",
    "remove_irrelevant_content", "strengthen_bullet"
]

VALID_TASKS = ["easy", "medium", "hard"]