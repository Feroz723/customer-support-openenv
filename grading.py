"""
Deterministic grading engine for Customer Support Resolution Environment.
All scoring is keyword-based — no LLM in the loop.
Same input → same output, always.
"""

from __future__ import annotations
from typing import Any
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

GENERIC_PHRASES = [
    "we will look into it",
    "apologies for the inconvenience",
    "apologize for the inconvenience",
    "working on it",
    "look into this for you",
]

# Security ordering keywords for hard_002
SECURITY_KEYWORDS = [
    "reset your password", "enable 2fa", "two-factor authentication",
    "secure your account", "password update", "freeze account",
]

REFUND_KEYWORDS = [
    "refund", "money back", "billing error", "double charge",
    "credit to your account",
]

SCORE_FLOOR = 0.01
SCORE_CEILING = 0.99


# ══════════════════════════════════════════════
# SCORING HELPERS
# ══════════════════════════════════════════════

def _contains_any(text: str, phrases: list[str]) -> bool:
    """Check if text contains any of the given phrases (case-insensitive)."""
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in phrases)


def _check_ordering(text: str, first_keywords: list[str], second_keywords: list[str]) -> bool:
    """
    Returns True if any keyword from second_keywords appears BEFORE first_keywords.
    Uses robust position index detection.
    """
    text_lower = text.lower()
    
    # Earliest position of a keyword from first_keywords
    first_pos = min([text_lower.find(k.lower()) for k in first_keywords if k.lower() in text_lower] or [float('inf')])
    
    # Earliest position of a keyword from second_keywords
    second_pos = min([text_lower.find(k.lower()) for k in second_keywords if k.lower() in text_lower] or [float('inf')])
    
    # Violation if both exist and second group (refund) comes before first group (security)
    if first_pos != float('inf') and second_pos != float('inf'):
        return second_pos < first_pos
    return False


# ══════════════════════════════════════════════
# DIMENSION SCORING
# ══════════════════════════════════════════════

def empathy_score(response: str, customer_name: str) -> tuple[float, str]:
    """Score empathy based on apology, acknowledgement, personalisation, closing."""
    score = 0.0
    details = []

    if _contains_any(response, APOLOGY_PHRASES):
        score += 0.10
        details.append("Apology")
    else:
        details.append("No apology")

    if _contains_any(response, ACKNOWLEDGEMENT_PHRASES):
        score += 0.10
        details.append("Acknowledgement")
    else:
        details.append("No acknowledgement")

    if customer_name.lower() in response.lower():
        score += 0.05
        details.append("Personalized")
    else:
        details.append("Not personalized")

    if _contains_any(response, POSITIVE_CLOSING_PHRASES):
        score += 0.05
        details.append("Positive closing")
    else:
        details.append("No closing")

    return round(score, 2), "; ".join(details)


def correctness_score(response: str, task: Task, ordering_violation: bool = False) -> tuple[float, str, int, int]:
    """Score based on resolutions matched (Ceiling: 0.50)."""
    rubric = task.rubric
    matched = 0
    total = len(rubric.expected_resolutions)
    details = []

    for res in rubric.expected_resolutions:
        keywords = rubric.resolution_keywords.get(res, [])
        if _contains_any(response, keywords):
            matched += 1
        else:
            details.append(f"Missed: {res}")

    base = 0.5 * (matched / total) if total > 0 else 0.0

    # 1. Ordering Deduction (-0.1)
    if ordering_violation:
        base -= 0.1
        details.append("Ordering failed")

    # 2. Sub-issue Deduction (-0.05 per miss)
    missed_subs = sum(1 for sub in rubric.sub_issues if sub.lower() not in response.lower())
    if rubric.sub_issues and missed_subs > len(rubric.sub_issues) // 2:
        base -= (0.05 * missed_subs)
        details.append(f"Sub-issue gaps")

    # 3. Final clamp to prevent distortion (Ceiling: 0.50, Floor: 0.0)
    final_base = max(0.0, min(base, 0.5))

    summary = f"Matched {matched}/{total}"
    if details:
        summary += f" ({details[0]})"

    return round(final_base, 2), summary, matched, total


def helpfulness_score(response: str) -> tuple[float, str]:
    """Score based on next steps, timeline, and follow-up (Ceiling: 0.2)."""
    score = 0.0
    details = []

    if _contains_any(response, NEXT_STEP_PHRASES):
        score += 0.07
        details.append("Steps")
    if _contains_any(response, TIMELINE_PHRASES):
        score += 0.07
        details.append("Timeline")
    if _contains_any(response, FOLLOWUP_PHRASES):
        score += 0.06
        details.append("Follow-up")

    return round(score, 2), "Provided: " + ", ".join(details) if details else "Lacks info"


def penalty_score(response: str, task: Task, cor_score: float, hlp_score: float, ordering_violation: bool = False) -> tuple[float, str]:
    """Apply penalties for low-effort, irrelevant, or flawed logic (Max Deduction: -0.30)."""
    penalty = 0.0
    details = []
    text_len = len(response.strip())

    # 1. Effort/Length
    if text_len < 50:
        penalty -= 0.1
        details.append("Too short")

    # 2. Forbidden phrases
    if task.rubric.forbidden_phrases and _contains_any(response, task.rubric.forbidden_phrases):
        penalty -= 0.1
        details.append("Forbidden phrases")

    # 3. Irrelevance
    all_keywords = [kw for sub in task.rubric.resolution_keywords.values() for kw in sub]
    if text_len > 20 and not _contains_any(response, all_keywords):
        penalty -= 0.2
        details.append("Irrelevant")

    # 4. Anti-Bluffing (Looks helpful but zero correctness)
    if cor_score < 0.2 and hlp_score > 0.15:
        penalty -= 0.1
        details.append("Bluffing check failed")

    # 5. Generic Fluff (No resolution keywords)
    if _contains_any(response, GENERIC_PHRASES) and cor_score < 0.1:
        penalty -= 0.1
        details.append("Generic fluff detected")

    # 6. Task-Specific Ordering Penalty (Crucial for hard_002)
    if ordering_violation:
        penalty -= 0.2  # Increased from -0.1 to punish priority error
        details.append("Priority violation")

    # 7. Consistency (Polite but empty responses on hard tasks)
    if task.observation.difficulty == "hard":
        if _contains_any(response, APOLOGY_PHRASES) and cor_score < 0.15:
            penalty -= 0.1
            details.append("Consistency check failed")

    # 8. Penalty Clamp (Prevent unrealistic collapse)
    final_penalty = max(-0.3, penalty)
    return round(final_penalty, 2), ", ".join(details) if details else "None"


# ══════════════════════════════════════════════
# MAIN GRADING FUNCTION
# ══════════════════════════════════════════════

def grade_response(response: str, task: Task) -> RewardBreakdown:
    """Grade response with strict calibration and robust ordering checks."""
    difficulty = task.observation.difficulty
    
    # 1. Early Ordering Detection (Security vs Refund)
    ordering_violation = False
    if task.task_id == "hard_002":
        ordering_violation = _check_ordering(response, SECURITY_KEYWORDS, REFUND_KEYWORDS)

    # 2. Calculate Dimension Scores
    emp_score, emp_det = empathy_score(response, task.observation.customer.name)
    cor_score, cor_det, matched, total_res = correctness_score(response, task, ordering_violation)
    hlp_score, hlp_det = helpfulness_score(response)
    
    # 3. Penalties (Include ordering and consistency)
    pen_score, pen_det = penalty_score(response, task, cor_score, hlp_score, ordering_violation)

    # Coverage Check (Matched < 50% resolutions)
    if total_res > 0 and matched < total_res / 2:
        pen_score = max(-0.3, pen_score - 0.1)
        pen_det += "; Coverage gap"

    # 4. Sum raw total
    total = emp_score + cor_score + hlp_score + pen_score

    # 5. Apply Gating (Throttlers)
    multiplier = 1.0
    gating_reason = ""

    # Correctness Gating (Primary Gate)
    if cor_score < 0.2:
        multiplier = min(multiplier, 0.5)
        gating_reason = "Correctness Gate (x0.5)"
    
    if cor_score < 0.1:
        multiplier = min(multiplier, 0.3)
        gating_reason = "Failure Gate (x0.3)"

    # Task/Security Booster for hard_002
    if task.task_id == "hard_002" and ordering_violation:
        multiplier = min(multiplier, 0.7)
        gating_reason = "Security Priority Gate (x0.7)"

    # General Hard/Medium Booster
    if difficulty == "hard" and cor_score < 0.3:
        multiplier = min(multiplier, 0.7)
        gating_reason = "Hard Task Requirement (x0.7)"
    
    if difficulty == "medium" and cor_score < 0.3:
        multiplier = min(multiplier, 0.8)
        gating_reason = "Medium Task Requirement (x0.8)"

    total *= multiplier

    # 6. Finalize Rewards
    # The evaluator requires scores to be strictly within (0, 1), never exactly 0 or 1.
    final_total = round(max(SCORE_FLOOR, min(SCORE_CEILING, total)), 2)

    # Build Structured Output
    reasoning = {
        "empathy": {"score": emp_score, "details": emp_det},
        "correctness": {"score": cor_score, "details": cor_det},
        "helpfulness": {"score": hlp_score, "details": hlp_det},
        "penalty": {"score": pen_score, "details": pen_det},
    }
    if gating_reason:
        reasoning["gating"] = {"multiplier": multiplier, "details": gating_reason}

    return RewardBreakdown(
        empathy_score=emp_score,
        correctness_score=cor_score,
        helpfulness_score=hlp_score,
        penalty=pen_score,
        total=final_total,
        reasoning=reasoning,
    )
