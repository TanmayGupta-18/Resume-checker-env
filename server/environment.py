# server/environment.py
import copy
import random
import re
from typing import Dict, Any, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# DATA — Resumes & Job Descriptions
# ══════════════════════════════════════════════════════════════════════════════

RESUMES = {
    "easy": {
        "name": "Priya Menon",
        "summary": "Final year CS student interested in machine learning and data science roles.",
        "skills": ["Python", "Machine Learning", "Pandas", "NumPy", "Git"],
        "experience": [
            {
                "title": "Data Science Intern",
                "company": "Infosys",
                "duration": "May 2023 - July 2023",
                "bullets": [
                    "Worked on customer churn prediction model",
                    "Helped clean and preprocess large datasets",
                    "Assisted team with model evaluation"
                ]
            }
        ],
        "education": {
            "degree": "B.Tech Computer Science",
            "college": "VIT Vellore",
            "gpa": 8.5,
            "year": 2024
        },
        "projects": [
            {
                "name": "Sentiment Analyzer",
                "description": "Built a sentiment analysis tool using Python and NLTK",
                "skills_used": ["Python", "NLTK", "Flask"]
            },
            {
                "name": "House Price Predictor",
                "description": "Predicted house prices using regression models",
                "skills_used": ["Python", "Scikit-learn", "Pandas"]
            }
        ]
    },
    "medium": {
        "name": "Rahul Verma",
        "summary": "I am a hardworking student who loves coding and wants to work in tech.",
        "skills": ["Python", "JavaScript", "some ML", "MySQL", "helped with Docker"],
        "experience": [
            {
                "title": "Intern",
                "company": "Startup XYZ",
                "duration": "Summer 2023",
                "bullets": [
                    "Did backend work",
                    "Was involved in making APIs",
                    "Helped the team with various tasks",
                    "Learned a lot about software development"
                ]
            }
        ],
        "education": {
            "degree": "B.E. Information Technology",
            "college": "Anna University",
            "gpa": 7.8,
            "year": 2024
        },
        "projects": [
            {
                "name": "Todo App",
                "description": "Made a todo app",
                "skills_used": ["JavaScript", "HTML", "CSS"]
            },
            {
                "name": "ML Project",
                "description": "Did a machine learning project for college",
                "skills_used": ["Python"]
            }
        ]
    },
    "hard": {
        "name": "Aditya Kumar",
        "summary": "Mechanical engineering student with interest in switching to software development.",
        "skills": ["AutoCAD", "SolidWorks", "MATLAB", "Basic Python", "MS Excel"],
        "experience": [
            {
                "title": "Manufacturing Intern",
                "company": "Tata Motors",
                "duration": "June 2023 - August 2023",
                "bullets": [
                    "Observed production line operations",
                    "Prepared reports on manufacturing efficiency",
                    "Used Excel to track inventory data"
                ]
            }
        ],
        "education": {
            "degree": "B.Tech Mechanical Engineering",
            "college": "NIT Trichy",
            "gpa": 7.2,
            "year": 2024
        },
        "projects": [
            {
                "name": "Robotic Arm Design",
                "description": "Designed a robotic arm using SolidWorks",
                "skills_used": ["SolidWorks", "MATLAB"]
            },
            {
                "name": "Python Automation Script",
                "description": "Wrote a Python script to automate Excel report generation",
                "skills_used": ["Python", "openpyxl"]
            }
        ]
    }
}

JOB_DESCRIPTIONS = {
    "easy": {
        "title": "Junior ML Engineer",
        "company": "Flipkart",
        "required_skills": ["Python", "Machine Learning", "Scikit-learn", "Pandas", "SQL"],
        "preferred_skills": ["TensorFlow", "Docker", "Git"],
        "keywords": ["machine learning", "model training", "data preprocessing", "feature engineering", "model evaluation", "python"],
        "weak_phrases_to_avoid": ["helped", "assisted", "worked on", "was involved in", "learned"],
        "strong_action_verbs": ["built", "developed", "implemented", "designed", "optimized", "reduced", "improved", "achieved", "deployed", "automated"],
        "experience_required": 0,
    },
    "medium": {
        "title": "Backend Software Engineer",
        "company": "Razorpay",
        "required_skills": ["Python", "REST APIs", "SQL", "Docker", "System Design"],
        "preferred_skills": ["Kubernetes", "Redis", "AWS", "Go"],
        "keywords": ["backend development", "REST API", "microservices", "database", "scalable systems", "software engineering", "agile"],
        "weak_phrases_to_avoid": ["helped", "assisted", "did", "was involved", "learned", "various tasks"],
        "strong_action_verbs": ["architected", "built", "scaled", "optimized", "reduced latency", "designed", "deployed", "implemented", "led", "automated"],
        "experience_required": 0,
    },
    "hard": {
        "title": "Software Development Engineer",
        "company": "Amazon",
        "required_skills": ["Python", "Data Structures", "Algorithms", "SQL", "OOP"],
        "preferred_skills": ["Java", "AWS", "System Design", "ML basics"],
        "keywords": ["software development", "algorithms", "data structures", "object oriented", "problem solving", "coding", "software engineering", "automation"],
        "weak_phrases_to_avoid": ["observed", "prepared reports", "basic", "interest in switching", "some experience", "helped"],
        "strong_action_verbs": ["automated", "developed", "engineered", "optimized", "built", "implemented", "designed", "solved", "reduced", "improved"],
        "experience_required": 0,
    }
}

ALL_ACTIONS = [
    "add_missing_keyword", "quantify_achievement", "remove_weak_phrase",
    "reorder_skills", "tailor_summary", "add_relevant_project",
    "remove_irrelevant_content", "strengthen_bullet",
]

# ══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_action(action_name: str, resume: dict, jd: dict) -> Tuple[dict, bool, str]:
    dispatch = {
        "add_missing_keyword": _add_missing_keyword,
        "quantify_achievement": _quantify_achievement,
        "remove_weak_phrase": _remove_weak_phrase,
        "reorder_skills": _reorder_skills,
        "tailor_summary": _tailor_summary,
        "add_relevant_project": _add_relevant_project,
        "remove_irrelevant_content": _remove_irrelevant_content,
        "strengthen_bullet": _strengthen_bullet,
    }
    if action_name not in dispatch:
        return resume, False, f"Unknown action: {action_name}"

    new_resume = copy.deepcopy(resume)
    success, message = dispatch[action_name](new_resume, jd)
    return new_resume, success, message


def _add_missing_keyword(resume, jd):
    current_skills = [s.lower() for s in resume["skills"]]
    for skill in jd["required_skills"] + jd["preferred_skills"]:
        if skill.lower() not in current_skills:
            resume["skills"].append(skill)
            return True, f"Added '{skill}' to skills"
    return False, "No new keywords to add"


def _quantify_achievement(resume, jd):
    for exp in resume["experience"]:
        for i, bullet in enumerate(exp["bullets"]):
            if not re.search(r'\d+', bullet):
                bullet_lower = bullet.lower()
                if "churn" in bullet_lower or "model" in bullet_lower:
                    exp["bullets"][i] = bullet.rstrip(".") + ", achieving 87% accuracy and reducing customer loss by 18%."
                elif "dataset" in bullet_lower or "clean" in bullet_lower:
                    exp["bullets"][i] = bullet.rstrip(".") + ", processing 650K+ records with 95% data quality."
                else:
                    exp["bullets"][i] = bullet.rstrip(".") + ", delivering 42% performance improvement and handling large-scale data."
                return True, "Quantified bullet successfully"
    return False, "No unquantified bullets left"


def _remove_weak_phrase(resume, jd):
    weak_map = {
        "helped": "Led", "assisted": "Drove", "worked on": "Developed",
        "did ": "Executed ", "was involved": "Owned", "learned": "Mastered",
        "observed": "Analyzed"
    }
    changed = False

    for exp in resume["experience"]:
        for i, bullet in enumerate(exp["bullets"]):
            new_bullet = bullet
            for weak, strong in weak_map.items():
                if weak.lower() in new_bullet.lower():
                    new_bullet = re.sub(re.escape(weak), strong, new_bullet, flags=re.IGNORECASE)
                    changed = True
            exp["bullets"][i] = new_bullet.strip()

    before = len(resume.get("skills", []))
    resume["skills"] = [s for s in resume.get("skills", [])
                        if not any(w in s.lower() for w in ["some", "basic", "helped with"])]
    if len(resume.get("skills", [])) < before:
        changed = True

    return changed, "Cleaned weak language with strong verbs" if changed else "No weak phrases remaining"


def _reorder_skills(resume, jd):
    required = [s.lower() for s in jd["required_skills"]]
    preferred = [s.lower() for s in jd["preferred_skills"]]
    top, mid, rest = [], [], []
    for skill in resume["skills"]:
        if skill.lower() in required:
            top.append(skill)
        elif skill.lower() in preferred:
            mid.append(skill)
        else:
            rest.append(skill)
    original = resume["skills"][:]
    resume["skills"] = top + mid + rest
    if original != resume["skills"]:
        return True, "Reordered skills: JD matches first"
    return False, "Skills already optimally ordered"


def _tailor_summary(resume, jd):
    if jd["title"].split()[0].lower() in resume["summary"].lower():
        return False, "Summary already mentions target role"
    top_skills = ", ".join(jd["required_skills"][:4])
    edu = resume["education"]
    resume["summary"] = (
        f"Final year {edu['degree']} student from {edu['college']} "
        f"seeking a {jd['title']} role at {jd['company']}. "
        f"Experienced in {top_skills} with hands-on project experience."
    )
    return True, f"Tailored summary for {jd['title']}"


def _add_relevant_project(resume, jd):
    if not resume.get("projects"):
        return False, "No projects available"
    if len(resume["projects"]) > 1:
        proj = resume["projects"].pop(0)
        resume["projects"].insert(0, proj)
        return True, "Moved relevant project to top"
    return False, "Projects already optimally ordered"


def _remove_irrelevant_content(resume, jd):
    irrelevant = ["autocad", "solidworks", "manufacturing", "mechanical"]
    before = len(resume["skills"])
    resume["skills"] = [s for s in resume["skills"] if not any(d in s.lower() for d in irrelevant)]
    if len(resume["skills"]) < before:
        return True, "Removed irrelevant skills"
    return False, "No irrelevant content found"


def _strengthen_bullet(resume, jd):
    weak_first = ["did", "made", "got", "saw", "used", "had", "was", "helped", "assisted"]
    verbs = jd.get("strong_action_verbs", ["Developed", "Implemented", "Built"])
    for exp in resume["experience"]:
        for i, bullet in enumerate(exp["bullets"]):
            first = bullet.split()[0].lower() if bullet.split() else ""
            if first in weak_first:
                verb = random.choice(verbs)
                rest = " ".join(bullet.split()[1:])
                exp["bullets"][i] = f"{verb} {rest}"
                return True, "Strengthened bullet opening"
    return False, "No weak opening verbs found"


# ══════════════════════════════════════════════════════════════════════════════
# STATE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_score(missing_kw, weak_count, quantified, total_bullets,
                   skill_ratio, tailored, irrelevant):
    kw_score      = max(0.0, 1.0 - len(missing_kw) / 5.0)
    quant_score   = min(1.0, (quantified / max(total_bullets, 1)) * 2.2)
    weak_score    = max(0.0, 1.0 - weak_count / max(total_bullets * 0.7, 1))
    skill_score   = min(1.0, skill_ratio * 1.7)
    summary_score = 1.0 if tailored else 0.6

    score = (
        0.28 * kw_score
      + 0.32 * quant_score
      + 0.20 * weak_score
      + 0.12 * skill_score
      + 0.13 * summary_score
      - 0.05 * min(irrelevant, 2)
    )
    return max(0.25, min(0.95, round(score, 3)))


def compute_state(resume: dict, jd: dict, steps_taken: int = 0, max_steps: int = 10, prev_resume: Optional[dict] = None) -> dict:
    missing_kw = _missing_keywords(resume, jd)
    weak_count = _weak_phrase_count(resume)
    quantified = _quantified_count(resume)
    total = _total_bullets(resume)
    skill_ratio = _skill_match_ratio(resume, jd)
    tailored = _summary_tailored(resume, jd)
    irrelevant = _irrelevant_skill_count(resume, jd)

    score = _compute_score(missing_kw, weak_count, quantified, total, skill_ratio, tailored, irrelevant)

    return {
        "current_score": round(score, 3),
        "steps_taken": steps_taken,
        "steps_remaining": max_steps - steps_taken,
        "missing_keywords": missing_kw,
        "weak_phrases_count": weak_count,
        "quantified_bullets": quantified,
        "total_bullets": total,
        "skill_match_ratio": round(skill_ratio, 3),
        "summary_tailored": tailored,
        "irrelevant_skills_count": irrelevant,
        "needs_keyword_work": len(missing_kw) > 0,
        "needs_quantification": quantified < total * 0.5,
        "needs_weak_cleanup": weak_count > 0,
        "needs_summary_work": not tailored,
        "needs_skill_reorder": skill_ratio < 0.8,
        "resume": resume,
        "job_description": jd,
        "resume_comparison": {"summary": "Changes applied", "num_changes": 1 if prev_resume else 0}
    }


def _missing_keywords(resume, jd):
    text = _flatten_text(resume).lower()
    return [s for s in jd["required_skills"] + jd["preferred_skills"] if s.lower() not in text]

def _weak_phrase_count(resume):
    weak = ["helped", "assisted", "worked on", "was involved", "did ", "learned", "observed"]
    return sum(1 for exp in resume.get("experience", []) for bullet in exp.get("bullets", []) if any(w in bullet.lower() for w in weak))

def _quantified_count(resume):
    return sum(1 for exp in resume.get("experience", []) for bullet in exp.get("bullets", []) if re.search(r'\d+', bullet))

def _total_bullets(resume):
    return sum(len(exp.get("bullets", [])) for exp in resume.get("experience", []))

def _skill_match_ratio(resume, jd):
    resume_skills = [s.lower() for s in resume.get("skills", [])]
    required = jd["required_skills"]
    return sum(1 for s in required if s.lower() in resume_skills) / len(required) if required else 1.0

def _summary_tailored(resume, jd):
    summary = resume.get("summary", "").lower()
    title_words = jd["title"].lower().split()
    return sum(1 for w in title_words if w in summary) >= 2

def _irrelevant_skill_count(resume, jd):
    irrelevant_domains = ["autocad", "solidworks", "manufacturing", "mechanical"]
    return sum(1 for s in resume.get("skills", []) if any(d in s.lower() for d in irrelevant_domains))

def _flatten_text(resume):
    parts = [resume.get("summary", ""), " ".join(resume.get("skills", []))]
    for exp in resume.get("experience", []):
        parts += [exp.get("title", "")] + exp.get("bullets", [])
    for proj in resume.get("projects", []):
        parts += [proj.get("name", ""), proj.get("description", "")]
        parts += proj.get("skills_used", [])
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ResumeOptimizationEnv:
    MAX_STEPS = 10  # fallback default

    def __init__(self, task: str = "easy", seed: int = None):
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"task must be easy | medium | hard, got '{task}'")
        self.task = task
        if seed is not None:
            random.seed(seed)
        self._resume = None
        self._jd = None
        self._steps_taken = 0
        self._initial_score = 0.0
        self._done = False
        # Per-task step limits
        self._max_steps = {"easy": 10, "medium": 15, "hard": 20}[task]

    def reset(self) -> dict:
        self._resume = copy.deepcopy(RESUMES[self.task])
        self._jd = copy.deepcopy(JOB_DESCRIPTIONS[self.task])
        self._steps_taken = 0
        self._done = False
        s = compute_state(self._resume, self._jd, 0, self._max_steps)
        self._initial_score = s["current_score"]
        return s

    def step(self, action: str) -> Tuple[dict, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode done — call reset() first")
        if action not in ALL_ACTIONS:
            raise ValueError(f"Invalid action '{action}'")

        prev_score = compute_state(self._resume, self._jd)["current_score"]
        new_resume, success, message = apply_action(action, self._resume, self._jd)
        self._resume = new_resume
        self._steps_taken += 1

        new_state = compute_state(self._resume, self._jd, self._steps_taken, self._max_steps)

        score_delta = new_state["current_score"] - prev_score
        reward = 1.75 * score_delta + (0.06 if success else 0.00)
        reward = round(max(0, min(0.70, reward)), 1)

        # done when score hits 0.95 OR task-specific step limit reached
        self._done = self._steps_taken >= self._max_steps or new_state["current_score"] >= 0.95

        info = {
            "action_success": success,
            "action_message": message,
            "score_before": prev_score,
            "score_after": new_state["current_score"],
        }
        return new_state, reward, self._done, info

    def state(self) -> dict:
        return compute_state(self._resume, self._jd, self._steps_taken, self._max_steps)

    def grade(self) -> float:
        final = self.state()["current_score"]
        improvement = final - self._initial_score
        return round(min(0.8 * final + 0.2 * min(improvement * 2, 1.0), 1.0), 3)
