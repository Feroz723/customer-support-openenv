"""
Pydantic data models for the Customer Support Resolution Environment.
Minimal MVP — single-turn, no FastAPI.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Any


# ──────────────────────────────────────────────
# Supporting models
# ──────────────────────────────────────────────

class CustomerProfile(BaseModel):
    name: str
    tier: Literal["standard", "premium", "enterprise"] = "standard"
    account_age_months: int = 12


class OrderDetails(BaseModel):
    order_id: str
    items: list[str]
    status: Literal["processing", "shipped", "delivered", "delayed", "cancelled"]
    expected_delivery: str          # human-readable date string
    tracking_number: str | None = None


class InteractionHistory(BaseModel):
    previous_tickets: int = 0
    escalation_count: int = 0


# ──────────────────────────────────────────────
# Core models
# ──────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees after reset() or step()."""
    ticket_id: str
    user_query: str
    sentiment: Literal["neutral", "frustrated", "angry", "anxious"]
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    customer: CustomerProfile
    order: OrderDetails | None = None
    history: InteractionHistory = Field(default_factory=InteractionHistory)
    internal_notes: list[str] = Field(default_factory=list)
    company_policies: dict[str, str] = Field(default_factory=dict)
    done: bool = False
    reward: float | None = None
    reward_breakdown: RewardBreakdown | None = None
    step_count: int = 0
    max_steps: int = 1


class Action(BaseModel):
    """What the agent sends to step()."""
    response: str = Field(..., min_length=1, max_length=8192)
    proposed_resolution: str | None = None
    escalate: bool = False


class RewardBreakdown(BaseModel):
    """Deterministic score breakdown returned after grading."""
    empathy_score: float = 0.0      # 0.0 – 0.3
    correctness_score: float = 0.0  # 0.0 – 0.4
    helpfulness_score: float = 0.0  # 0.0 – 0.3
    penalty: float = 0.0            # 0.0 to -0.2
    total: float = 0.01              # clamped (0, 1) — never exactly 0 or 1
    reasoning: dict[str, Any] = Field(default_factory=dict)


# Pydantic needs RewardBreakdown defined before Observation references it,
# but we used a forward ref above — rebuild to resolve.
Observation.model_rebuild()
