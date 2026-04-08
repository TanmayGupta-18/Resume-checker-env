# client.py
# HTTPEnvClient — what agents import to interact with the environment
# Hides all HTTP calls behind a clean reset() / step() / state() interface

import requests
from models import Observation, StepResponse, GradeResponse, VALID_ACTIONS


class HTTPEnvClient:
    """
    Client for the Resume Optimization Environment server.

    Usage:
        client = HTTPEnvClient("http://localhost:7860")
        obs    = client.reset(task="medium", seed=42)

        done = False
        while not done:
            action   = your_agent.select_action(obs)
            obs, reward, done, info = client.step(action)

        grade = client.grade()
        print(f"Final grade: {grade.grade:.3f}")
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to environment server at {self.base_url}. "
                f"Start the server with: uvicorn server.app:app --port 7860"
            )

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, task: str = "easy", seed: int = None) -> Observation:
        """
        Start a new episode.

        Args:
            task: "easy" | "medium" | "hard"
            seed: Integer for reproducibility (optional)

        Returns:
            Observation — initial state of the environment
        """
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed

        r = requests.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return Observation(**r.json())

    def step(self, action: str) -> tuple[Observation, float, bool, dict]:
        """
        Take one action in the environment.

        Args:
            action: One of VALID_ACTIONS

        Returns:
            (observation, reward, done, info)
        """
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid: {VALID_ACTIONS}")

        r = requests.post(f"{self.base_url}/step", json={"name": action})
        r.raise_for_status()

        data = StepResponse(**r.json())
        return data.observation, data.reward, data.done, data.info

    def state(self) -> Observation:
        """Get current observation without acting."""
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return Observation(**r.json())

    def grade(self) -> GradeResponse:
        """Get final grade for the current episode (0.0 - 1.0)."""
        r = requests.get(f"{self.base_url}/grade")
        r.raise_for_status()
        return GradeResponse(**r.json())

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def action_space(self) -> list[str]:
        return VALID_ACTIONS

    def run_episode(self, agent, task: str = "easy", seed: int = 42, verbose: bool = True):
        """
        Convenience method — run a full episode with any agent.

        Args:
            agent: Object with a .select_action(observation) -> str method
            task:  "easy" | "medium" | "hard"
            seed:  For reproducibility
            verbose: Print step-by-step progress

        Returns:
            GradeResponse with final grade
        """
        obs  = self.reset(task=task, seed=seed)
        done = False
        step = 0

        if verbose:
            print(f"\nTask: {task.upper()} | "
                  f"Candidate: {obs.resume['name']} → "
                  f"{obs.job_description['title']} @ {obs.job_description['company']}")
            print(f"Initial score: {obs.current_score:.3f}\n")

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = self.step(action)
            step += 1

            if verbose:
                status = "✓" if info["action_success"] else "✗"
                print(f"Step {step:2d} | {action:<30} | "
                      f"Reward: {reward:+.3f} | "
                      f"Score: {obs.current_score:.3f} | "
                      f"{status} {info['action_message'][:45]}")

        result = self.grade()
        if verbose:
            print(f"\nFinal grade: {result.grade:.3f}")
        return result