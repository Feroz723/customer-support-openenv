"""
Customer Support Resolution Environment — OpenEnv compatible.
Implements: reset(), step(), state(), close()
Single-turn, deterministic grading, no server required.
"""

from __future__ import annotations
import uuid
import copy

from models import Observation, Action, RewardBreakdown
from tasks import Task, get_task, get_task_by_difficulty, list_task_ids
from grading import grade_response


class CustomerSupportEnv:
    """
    OpenEnv-compatible environment for customer support resolution.

    Usage:
        env = CustomerSupportEnv()
        obs = env.reset()                        # or env.reset(task_id="easy_001")
        obs = env.step(Action(response="..."))   # grades response, returns reward
        state = env.state()                      # metadata + history
    """

    def __init__(self):
        self._episode_id: str | None = None
        self._task: Task | None = None
        self._observation: Observation | None = None
        self._step_count: int = 0
        self._max_steps: int = 1        # single-turn
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._history: list[dict] = []  # list of {observation, action, reward}
        self._reward_breakdown: RewardBreakdown | None = None

    # ──────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────

    def reset(self, task_id: str | None = None, difficulty: str | None = None) -> Observation:
        """
        Start a new episode.

        Args:
            task_id:    Specific task ID (e.g. "easy_001"). Takes priority.
            difficulty: Pick first task of this difficulty ("easy"/"medium"/"hard").
                        Ignored if task_id is provided.

        Returns:
            Initial Observation (done=False, reward=None).
        """
        # Load the task
        if task_id:
            self._task = get_task(task_id)
        elif difficulty:
            self._task = get_task_by_difficulty(difficulty)
        else:
            # Default to easy
            self._task = get_task("easy_001")

        # Reset episode state
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history = []
        self._reward_breakdown = None

        # Build initial observation (deep copy so task data stays clean)
        self._observation = copy.deepcopy(self._task.observation)
        self._observation.done = False
        self._observation.reward = 0.01
        self._observation.reward_breakdown = RewardBreakdown()
        self._observation.step_count = 0
        self._observation.max_steps = self._max_steps

        return self._observation

    # ──────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────

    def step(self, action: Action) -> Observation:
        """
        Execute the agent's action (response) and return graded observation.

        Args:
            action: Action containing the agent's text response.

        Returns:
            Observation with reward filled in and done=True (single-turn).

        Raises:
            RuntimeError: If episode hasn't been started or is already done.
        """
        if self._task is None or self._episode_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Grade the response
        reward_breakdown = grade_response(action.response, self._task)

        # Update state
        self._step_count += 1
        self._cumulative_reward += reward_breakdown.total
        self._done = self._step_count >= self._max_steps
        self._reward_breakdown = reward_breakdown

        # Save to history
        self._history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward_breakdown.model_dump(),
        })

        # Build result observation
        obs = copy.deepcopy(self._task.observation)
        obs.done = self._done
        obs.reward = reward_breakdown.total
        obs.reward_breakdown = reward_breakdown
        obs.step_count = self._step_count
        obs.max_steps = self._max_steps
        self._observation = obs

        return obs

    # ──────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────

    def state(self) -> dict:
        """
        Return current environment state / metadata.

        Returns:
            Dict with episode info, step count, reward, history.
        """
        return {
            "episode_id": self._episode_id,
            "task_id": self._task.task_id if self._task else None,
            "difficulty": self._task.observation.difficulty if self._task else None,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 2),
            "reward_breakdown": (
                self._reward_breakdown.model_dump() if self._reward_breakdown else None
            ),
            "history": self._history,
            "available_tasks": list_task_ids(),
        }

    # ──────────────────────────────────────────
    # close()
    # ──────────────────────────────────────────

    def close(self):
        """Clean up resources (no-op for now)."""
        self._episode_id = None
        self._task = None
        self._observation = None
        self._done = True
        self._history = []


# ══════════════════════════════════════════════
# Quick smoke test (run this file directly)
# ══════════════════════════════════════════════

if __name__ == "__main__":
    env = CustomerSupportEnv()

    print("=" * 60)
    print("CUSTOMER SUPPORT RESOLUTION ENVIRONMENT — SMOKE TEST")
    print("=" * 60)

    for task_id in list_task_ids():
        obs = env.reset(task_id=task_id)
        print(f"\n{'─' * 60}")
        print(f"Task: {task_id} | Difficulty: {obs.difficulty}")
        print(f"Customer: {obs.customer.name} ({obs.customer.tier})")
        print(f"Sentiment: {obs.sentiment}")
        print(f"Query: {obs.user_query[:100]}...")

        # Simulate a decent agent response
        if task_id == "easy_001":
            response = (
                "Dear James, I sincerely apologize for the delay with your order #ORD-90412. "
                "I understand your frustration — this should not have happened. The delay is due "
                "to a supplier backlog at our warehouse. Your Bluetooth speaker is expected to "
                "ship by March 30th, and you'll receive a tracking number via email once it does. "
                "As a gesture of goodwill, I'll also send you a 10% discount code for your next "
                "purchase. Please don't hesitate to reach out if you need anything else. "
                "Thank you for your patience!"
            )
        elif task_id == "med_001":
            response = (
                "Dear Priya, I sincerely apologize for the frustrating experience with order "
                "#ORD-78234. I completely understand your frustration — waiting this long is "
                "unacceptable. Here's what I'll do for you right away: I'm processing a full "
                "refund for the ErgoFlex Laptop Stand immediately. As a premium member, your "
                "refund will be processed within 2-3 business days. For the NovaDock USB-C Hub, "
                "I'm arranging priority shipping today — it will ship within 1 business day and "
                "you'll receive a tracking number via email. I'll personally follow up to make "
                "sure everything arrives on time. Thank you for your patience, Priya."
            )
        else:
            response = (
                "Dear Marcus, I am deeply sorry for this completely unacceptable experience. "
                "I want to acknowledge that we have failed you — this should have been resolved "
                "after your first contact, and the fact that previous agents did not process your "
                "replacement is inexcusable. Here's what I'm doing right now: First, I'm processing "
                "a full refund for the damaged NovaPro Wireless Headphones immediately. Second, I'm "
                "personally arranging an overnight replacement to ship today. Third, I'm adding a "
                "$25 store credit to your account as compensation for the time and frustration this "
                "has caused. I'm also escalating your case to our senior support team to ensure "
                "a dedicated agent follows up with you within 24 hours. You are a valued customer, "
                "Marcus, and I will personally ensure this gets resolved today. Please don't hesitate "
                "to contact us directly if you need anything else."
            )

        result = env.step(Action(response=response))

        print(f"\n  Reward: {result.reward}")
        print(f"  Empathy:      {result.reward_breakdown.empathy_score}/0.30")
        print(f"  Correctness:  {result.reward_breakdown.correctness_score}/0.40")
        print(f"  Helpfulness:  {result.reward_breakdown.helpfulness_score}/0.30")
        print(f"  Penalty:      {result.reward_breakdown.penalty}")
        print(f"  TOTAL:        {result.reward_breakdown.total}/1.00")
        print(f"  Done:         {result.done}")

    # Show final state
    print(f"\n{'=' * 60}")
    print("FINAL STATE:")
    state = env.state()
    print(f"  Episode: {state['episode_id']}")
    print(f"  Task:    {state['task_id']}")
    print(f"  Steps:   {state['step_count']}/{state['max_steps']}")
    print(f"  Reward:  {state['cumulative_reward']}")
    print(f"  Tasks:   {state['available_tasks']}")

    env.close()
    print("\n✅ Environment closed. Smoke test complete.")
