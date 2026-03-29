"""
Deterministic grading engine for Customer Support Resolution Environment.
All scoring is keyword-based — no LLM in the loop.
Same input → same output, always.
"""

from __future__ import annotations
from models import RewardBreakdown
from tasks import Task


# ══════════════════════════════════════════════
# KEYWORD LISTS
# ══════════════════════════════════════════════

APOLOGY_PHRASES = [
    "sorry", "apologize", "apologies", "regret",
    "sincerely sorry", "deeply sorry", "truly sorry",
    "deeply apologize",
]

ACKNOWLEDGEMENT_PHRASES = [
    "understand your frustration",
    "appreciate your patience",
    "understand how frustrating",
    "completely understand",
    "can see how",
    "must be frustrating",
    "can imagine how",
    "hear you",
]

POSITIVE_CLOSING_PHRASES = [
    "thank you for your patience",
    "valued customer",
    "anything else",
    "happy to help",
    "don't hesitate",
    "here for you",
    "glad to assist",
    "pleasure to help",
]

NEXT_STEP_PHRASES = [
    "here's what", "here is what", "next step",
    "what happens next", "you can expect",
    "we will", "i will", "i'll",
    "please follow", "steps to",
    "going to", "plan is to",
]

TIMELINE_PHRASES = [
    "within", "business days", "hours",
    "by march", "by april", "estimated",
    "expected to", "should arrive", "will be processed",
    "working days", "right away", "immediately",
    "today", "tomorrow", "overnight",
]

FOLLOWUP_PHRASES = [
    "follow up", "follow-up", "reach out",
    "contact us", "call us", "email us",
    "get back to you", "update you",
    "keep you posted", "let you know",
    "tracking number", "confirmation email",
]


# ══════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════

def _contains_any(text: str, phrases: list[str]) -> bool:
    """Check if text contains any of the given phrases (case-insensitive)."""
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in phrases)


def _count_matches(text: str, phrases: list[str]) -> int:
    """Count how many phrases are found in the text."""
    text_lower = text.lower()
    return sum(1 for phrase in phrases if phrase.lower() in text_lower)


# ──────────────────────────────────────────────
# Empathy (0.0 – 0.3)
# ──────────────────────────────────────────────

def empathy_score(response: str, customer_name: str) -> tuple[float, dict[str, str]]:
    """Score empathy based on apology, acknowledgement, personalisation, closing."""
    score = 0.0
    reasons: dict[str, str] = {}

    # Apology present → +0.10
    if _contains_any(response, APOLOGY_PHRASES):
        score += 0.10
        reasons["apology"] = "Apology phrase detected"
    else:
        reasons["apology"] = "No apology found"

    # Acknowledges frustration → +0.10
    if _contains_any(response, ACKNOWLEDGEMENT_PHRASES):
        score += 0.10
        reasons["acknowledgement"] = "Frustration acknowledged"
    else:
        reasons["acknowledgement"] = "No acknowledgement of frustration"

    # Uses customer name → +0.05
    if customer_name.lower() in response.lower():
        score += 0.05
        reasons["personalisation"] = f"Customer name '{customer_name}' used"
    else:
        reasons["personalisation"] = "Customer name not used"

    # Positive closing → +0.05
    if _contains_any(response, POSITIVE_CLOSING_PHRASES):
        score += 0.05
        reasons["closing"] = "Positive closing detected"
    else:
        reasons["closing"] = "No positive closing"

    return round(score, 2), reasons


# ──────────────────────────────────────────────
# Correctness (0.0 – 0.4)
# ──────────────────────────────────────────────

def correctness_score(response: str, task: Task) -> tuple[float, dict[str, str]]:
    """Score based on whether the response addresses expected resolutions."""
    rubric = task.rubric
    reasons: dict[str, str] = {}
    matched = 0
    total = len(rubric.expected_resolutions)

    for resolution_key in rubric.expected_resolutions:
        keywords = rubric.resolution_keywords.get(resolution_key, [])
        if _contains_any(response, keywords):
            matched += 1
            reasons[resolution_key] = "✓ Addressed"
        else:
            reasons[resolution_key] = "✗ Not addressed"

    # Base score: proportional to resolutions matched
    base = 0.4 * (matched / total) if total > 0 else 0.0

    # Deduction for missed sub-issues (−0.03 each, but don't go below 0)
    missed_subs = 0
    for sub in rubric.sub_issues:
        if sub.lower() not in response.lower():
            missed_subs += 1

    # Only penalise if there are sub_issues defined and most were missed
    if rubric.sub_issues and missed_subs > len(rubric.sub_issues) // 2:
        deduction = 0.03 * missed_subs
        base = max(0.0, base - deduction)
        reasons["sub_issues"] = f"Missed {missed_subs}/{len(rubric.sub_issues)} sub-issues"

    reasons["resolution_match"] = f"{matched}/{total} resolutions addressed"
    return round(base, 2), reasons


# ──────────────────────────────────────────────
# Helpfulness (0.0 – 0.3)
# ──────────────────────────────────────────────

def helpfulness_score(response: str) -> tuple[float, dict[str, str]]:
    """Score based on next steps, timeline, and follow-up information."""
    score = 0.0
    reasons: dict[str, str] = {}

    # Clear next steps → +0.10
    if _contains_any(response, NEXT_STEP_PHRASES):
        score += 0.10
        reasons["next_steps"] = "Next steps provided"
    else:
        reasons["next_steps"] = "No clear next steps"

    # Timeline / ETA → +0.10
    if _contains_any(response, TIMELINE_PHRASES):
        score += 0.10
        reasons["timeline"] = "Timeline / ETA provided"
    else:
        reasons["timeline"] = "No timeline given"

    # Follow-up / contact → +0.10
    if _contains_any(response, FOLLOWUP_PHRASES):
        score += 0.10
        reasons["followup"] = "Follow-up / contact info provided"
    else:
        reasons["followup"] = "No follow-up offered"

    return round(score, 2), reasons


# ──────────────────────────────────────────────
# Penalty (0.0 to −0.2)
# ──────────────────────────────────────────────

def penalty_score(response: str, task: Task) -> tuple[float, dict[str, str]]:
    """Apply penalties for low-effort, irrelevant, or rude responses."""
    penalty = 0.0
    reasons: dict[str, str] = {}

    # Too short (< 50 chars) → −0.10
    if len(response.strip()) < 50:
        penalty -= 0.10
        reasons["too_short"] = f"Response too short ({len(response.strip())} chars)"

    # Forbidden phrases → −0.10
    rubric = task.rubric
    if rubric.forbidden_phrases and _contains_any(response, rubric.forbidden_phrases):
        penalty -= 0.10
        reasons["forbidden"] = "Contains forbidden/dismissive phrasing"

    # Completely irrelevant (doesn't mention ANY key terms from the task) → −0.20
    all_keywords = []
    for kw_list in rubric.resolution_keywords.values():
        all_keywords.extend(kw_list)

    if not _contains_any(response, all_keywords) and len(response.strip()) > 20:
        penalty -= 0.20
        reasons["irrelevant"] = "Response appears completely off-topic"

    # Cap at -0.2
    penalty = max(-0.2, penalty)
    if not reasons:
        reasons["penalty"] = "No penalties applied"

    return round(penalty, 2), reasons


# ══════════════════════════════════════════════
# MAIN GRADING FUNCTION
# ══════════════════════════════════════════════

def grade_response(response: str, task: Task) -> RewardBreakdown:
    """
    Grade an agent's response against a task rubric.
    Returns a RewardBreakdown with scores per dimension and total.
    Fully deterministic — same input always produces same output.
    """
    customer_name = task.observation.customer.name

    emp_score, emp_reasons = empathy_score(response, customer_name)
    cor_score, cor_reasons = correctness_score(response, task)
    hlp_score, hlp_reasons = helpfulness_score(response)
    pen_score, pen_reasons = penalty_score(response, task)

    total = emp_score + cor_score + hlp_score + pen_score
    total = round(max(0.0, min(1.0, total)), 2)

    # Merge all reasoning
    reasoning = {}
    reasoning.update({f"empathy_{k}": v for k, v in emp_reasons.items()})
    reasoning.update({f"correctness_{k}": v for k, v in cor_reasons.items()})
    reasoning.update({f"helpfulness_{k}": v for k, v in hlp_reasons.items()})
    reasoning.update({f"penalty_{k}": v for k, v in pen_reasons.items()})

    return RewardBreakdown(
        empathy_score=emp_score,
        correctness_score=cor_score,
        helpfulness_score=hlp_score,
        penalty=pen_score,
        total=total,
        reasoning=reasoning,
    )
