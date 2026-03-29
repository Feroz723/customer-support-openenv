"""
Task definitions for the Customer Support Resolution Environment.
Contains 4 tasks (Easy, Medium, Hard, Hard-Security) for evaluation.
Company: NovaMart (Fictional E-commerce Retailer).
"""

from __future__ import annotations
from models import (
    Observation, CustomerProfile, OrderDetails, InteractionHistory,
)


# ──────────────────────────────────────────────
# Grading rubric per task
# ──────────────────────────────────────────────

class TaskRubric:
    """Defines what the grader looks for in a correct response."""

    def __init__(
        self,
        expected_resolutions: list[str],
        resolution_keywords: dict[str, list[str]],
        sub_issues: list[str] | None = None,
        forbidden_phrases: list[str] | None = None,
    ):
        self.expected_resolutions = expected_resolutions
        self.resolution_keywords = resolution_keywords
        self.sub_issues = sub_issues or []
        self.forbidden_phrases = forbidden_phrases or []


# ──────────────────────────────────────────────
# Task container
# ──────────────────────────────────────────────

class Task:
    def __init__(self, task_id: str, observation: Observation, rubric: TaskRubric):
        self.task_id = task_id
        self.observation = observation
        self.rubric = rubric


# ══════════════════════════════════════════════
# TASK DEFINITIONS
# ══════════════════════════════════════════════

TASKS: dict[str, Task] = {}

# ──────────────────────────────────────────────
# EASY — Delayed Order
# ──────────────────────────────────────────────

TASKS["easy_001"] = Task(
    task_id="easy_001",
    observation=Observation(
        ticket_id="TKT-20260328-0001",
        user_query=(
            "Hi, I placed an order 5 days ago (Order #ORD-90412) for a Bluetooth speaker "
            "and it was supposed to arrive by March 25th, but it still hasn't shipped. "
            "Can you tell me what's going on and when I'll get it?"
        ),
        sentiment="frustrated",
        category="delayed_order",
        difficulty="easy",
        customer=CustomerProfile(
            name="James Porter",
            tier="standard",
            account_age_months=8,
        ),
        order=OrderDetails(
            order_id="ORD-90412",
            items=["Bluetooth Speaker – NovaBass Pro"],
            status="delayed",
            expected_delivery="March 25, 2026",
            tracking_number=None,
        ),
        history=InteractionHistory(previous_tickets=0, escalation_count=0),
        internal_notes=[
            "Warehouse backlog due to supplier delay. Estimated restock: March 30.",
            "Tracking number will be assigned once item ships.",
        ],
        company_policies={
            "shipping_delay": (
                "If an order is delayed beyond the estimated delivery date, "
                "NovaMart will provide a revised delivery estimate and a 10% discount "
                "code for the next purchase."
            ),
        },
    ),
    rubric=TaskRubric(
        expected_resolutions=[
            "APOLOGIZE_DELAY",
            "EXPLAIN_REASON",
            "GIVE_TIMELINE",
        ],
        resolution_keywords={
            "APOLOGIZE_DELAY": [
                "sorry", "apologize", "apologies", "regret",
                "sincerely sorry", "deeply sorry",
            ],
            "EXPLAIN_REASON": [
                "supplier delay", "warehouse", "backlog",
                "restock", "delay in processing",
            ],
            "GIVE_TIMELINE": [
                "march 30", "within 5 days", "business days",
                "estimated", "expected to ship", "revised delivery",
            ],
        },
    ),
)

# ──────────────────────────────────────────────
# MEDIUM — Delayed Order + Refund Request
# ──────────────────────────────────────────────

TASKS["med_001"] = Task(
    task_id="med_001",
    observation=Observation(
        ticket_id="TKT-20260328-0002",
        user_query=(
            "I ordered a laptop stand and a USB-C hub two weeks ago (Order #ORD-78234). "
            "The delivery date was March 18th. It's now March 28th and I still haven't received anything. "
            "I've been waiting too long — I want a full refund for the laptop stand, "
            "but I still need the USB-C hub so please ship that as soon as possible. "
            "This is really frustrating."
        ),
        sentiment="frustrated",
        category="delayed_order",
        difficulty="medium",
        customer=CustomerProfile(
            name="Priya Sharma",
            tier="premium",
            account_age_months=26,
        ),
        order=OrderDetails(
            order_id="ORD-78234",
            items=["ErgoFlex Laptop Stand", "NovaDock USB-C Hub"],
            status="delayed",
            expected_delivery="March 18, 2026",
            tracking_number=None,
        ),
        history=InteractionHistory(previous_tickets=1, escalation_count=0),
        internal_notes=[
            "Laptop stand: out of stock, no restock ETA.",
            "USB-C hub: in stock, can ship within 1 business day.",
            "Customer is premium tier — eligible for priority shipping at no extra cost.",
        ],
        company_policies={
            "refund_policy": (
                "Full refunds are processed within 5-7 business days to the original "
                "payment method. Premium members receive expedited 2-3 business day refunds."
            ),
            "partial_order": (
                "If only part of an order is available, NovaMart can split-ship the "
                "available item(s) and process a refund for unavailable items."
            ),
        },
    ),
    rubric=TaskRubric(
        expected_resolutions=[
            "APOLOGIZE_DELAY",
            "CONFIRM_REFUND",
            "REFUND_TIMELINE",
            "SHIP_REMAINING",
        ],
        resolution_keywords={
            "APOLOGIZE_DELAY": [
                "sorry", "apologize", "apologies", "regret",
                "understand your frustration",
            ],
            "CONFIRM_REFUND": [
                "full refund", "refund for the laptop stand",
                "process a refund", "refund will be issued",
                "refund the laptop stand",
            ],
            "REFUND_TIMELINE": [
                "2-3 business days", "5-7 business days",
                "within", "business days", "refund timeline",
                "processed within",
            ],
            "SHIP_REMAINING": [
                "ship the usb-c hub", "ship the hub",
                "expedited shipping", "priority shipping",
                "ship within 1 business day", "send the usb",
                "ship the remaining", "dispatch",
            ],
        },
        sub_issues=["delayed_order", "refund_request", "partial_shipment"],
    ),
)

# ──────────────────────────────────────────────
# HARD — Repeated Service Failure (hard_001)
# ──────────────────────────────────────────────

TASKS["hard_001"] = Task(
    task_id="hard_001",
    observation=Observation(
        ticket_id="TKT-20260328-0003",
        user_query=(
            "I am absolutely FURIOUS. This is the THIRD time I'm contacting NovaMart about "
            "order #ORD-55190. First, my package was delayed by two weeks. Then when it "
            "finally arrived, the NovaPro Wireless Headphones were DAMAGED — cracked headband, "
            "left ear cup not working. I requested a replacement 10 DAYS AGO and nobody has "
            "done ANYTHING. No replacement, no refund, no update. I've been a loyal customer "
            "for 3 years and this is how you treat me? I want a FULL REFUND immediately, "
            "a replacement sent overnight, AND some kind of compensation for wasting my time. "
            "If this isn't resolved TODAY I'm disputing with my bank and leaving a review "
            "everywhere. This is unacceptable."
        ),
        sentiment="angry",
        category="delayed_order",
        difficulty="hard",
        customer=CustomerProfile(
            name="Marcus Williams",
            tier="premium",
            account_age_months=36,
        ),
        order=OrderDetails(
            order_id="ORD-55190",
            items=["NovaPro Wireless Headphones"],
            status="delivered",
            expected_delivery="March 5, 2026",
            tracking_number="NVM-TRK-881204",
        ),
        history=InteractionHistory(previous_tickets=3, escalation_count=2),
        internal_notes=[
            "Customer has contacted 3 times. Previous agents promised replacement but it was never processed.",
            "Replacement unit is in stock and can ship overnight.",
            "Customer is premium tier for 3 years — high retention value.",
            "Authorised: full refund OR overnight replacement + $25 store credit as goodwill.",
            "If customer threatens chargeback, escalate to retention team immediately.",
        ],
        company_policies={
            "damaged_item": (
                "Damaged items are eligible for free replacement or full refund. "
                "NovaMart covers return shipping costs."
            ),
            "compensation": (
                "For repeated service failures, NovaMart may offer store credit "
                "up to $25 or a discount code for future purchases as goodwill compensation."
            ),
            "escalation": (
                "If a customer has contacted support 3 or more times for the same issue "
                "without resolution, the case should be escalated to a senior support agent."
            ),
        },
    ),
    rubric=TaskRubric(
        expected_resolutions=[
            "APOLOGIZE_REPEATED",
            "ACKNOWLEDGE_FAILURES",
            "FULL_REFUND_OR_REPLACEMENT",
            "COMPENSATION",
            "ESCALATION_OFFER",
        ],
        resolution_keywords={
            "APOLOGIZE_REPEATED": [
                "sincerely sorry", "deeply apologize", "truly sorry",
                "unacceptable", "should not have happened",
                "understand your frustration", "apologize for",
            ],
            "ACKNOWLEDGE_FAILURES": [
                "dropped the ball", "failed you", "let you down",
                "previous agents", "should have been resolved",
                "fell through", "our mistake", "our fault",
                "inexcusable", "acknowledge",
            ],
            "FULL_REFUND_OR_REPLACEMENT": [
                "full refund", "overnight replacement", "overnight shipping",
                "send a replacement", "replacement immediately",
                "express ship", "refund will be processed",
                "ship right away", "replacement today",
            ],
            "COMPENSATION": [
                "store credit", "$25", "discount", "compensation",
                "goodwill", "complimentary", "credit to your account",
                "gift card", "courtesy",
            ],
            "ESCALATION_OFFER": [
                "escalat", "senior agent", "supervisor",
                "dedicated support", "retention team",
                "personal follow-up", "personally ensure",
                "direct contact", "case manager",
            ],
        },
        sub_issues=[
            "shipping_delay", "damaged_item", "replacement_not_processed",
            "repeated_contact", "chargeback_threat",
        ],
        forbidden_phrases=[
            "that's not our fault",
            "nothing we can do",
            "you should have",
            "it's your responsibility",
            "per our policy we cannot",
        ],
    ),
)


# ──────────────────────────────────────────────
# HARD — Security Breach & Billing Error (hard_002)
# ──────────────────────────────────────────────

TASKS["hard_002"] = Task(
    task_id="hard_002",
    observation=Observation(
        ticket_id="TKT-20260328-0004",
        user_query=(
            "I checked my account this morning and I'm LIVID. There was an unauthorized "
            "login from a location I don't recognize, and now I see a DOUBLE charge of $149.99 "
            "for my recent order #ORD-44912. This is a massive security failure! "
            "I want my money back for BOTH charges immediately and my account deleted. "
            "How could you let this happen? I don't feel safe using NovaMart anymore."
        ),
        sentiment="angry",
        category="security_billing",
        difficulty="hard",
        customer=CustomerProfile(
            name="Sarah Jenkins",
            tier="enterprise",
            account_age_months=48,
        ),
        order=OrderDetails(
            order_id="ORD-44912",
            items=["NovaHome Smart Security Hub"],
            status="delivered",
            expected_delivery="March 20, 2026",
            tracking_number="NVM-TRK-990123",
        ),
        history=InteractionHistory(previous_tickets=0, escalation_count=0),
        internal_notes=[
            "Confirmed unauthorized login from unknown IP yesterday.",
            "Double charge: system error during payment processing. Refund authorized for ONE charge.",
            "DO NOT delete account immediately if possible; try to secure it first.",
            "Policy: Data breach requires password reset and 2FA activation before any financial settlements.",
            "Authorized: $50 'Peace of Mind' credit if security steps are completed.",
        ],
        company_policies={
            "security_breach": (
                "In cases of unauthorized access, the agent MUST first instruct the "
                "customer to reset their password and enable Two-Factor Authentication (2FA). "
                "Account billing issues will be addressed ONLY after the account is secured."
            ),
            "refund_billing": (
                "Duplicate charges are investigated and refunded within 3-5 business days. "
                "Instant refunds are NOT permitted for security-related cases."
            ),
        },
    ),
    rubric=TaskRubric(
        expected_resolutions=[
            "APOLOGIZE_SECURITY",
            "SECURITY_ACTION",
            "ACKNOWLEDGE_BILLING",
            "EXPLAIN_INVESTIGATION",
            "COMPENSATION",
        ],
        resolution_keywords={
            "APOLOGIZE_SECURITY": [
                "security concern", "unauthorized access", "deeply apologize",
                "take security seriously", "safety is our priority",
                "understand your alarm", "sorry for the breach",
            ],
            "SECURITY_ACTION": [
                "reset your password", "enable 2fa", "two-factor authentication",
                "secure your account", "immediately reset", "password update",
                "freeze account",
            ],
            "ACKNOWLEDGE_BILLING": [
                "double charge", "$149.99", "duplicate payment",
                "erroneous charge", "billing error",
            ],
            "EXPLAIN_INVESTIGATION": [
                "investigate the charge", "3-5 business days", "review the duplicate",
                "not an instant refund", "requires a review", "refund timeline",
            ],
            "COMPENSATION": [
                "$50", "peace of mind credit", "store credit", "goodwill",
                "compensation", "apology credit",
            ],
        },
        sub_issues=[
            "security_breach", "double_charge", "account_deletion_request",
            "compliance",
        ],
        forbidden_phrases=[
            "not a big deal",
            "you must have shared your password",
            "we can't help with security",
            "just a glitch",
        ],
    ),
)


# ──────────────────────────────────────────────
# Registry helpers
# ──────────────────────────────────────────────

def get_task(task_id: str) -> Task:
    """Return a task by ID. Raises KeyError if not found."""
    if task_id not in TASKS:
        raise KeyError(
            f"Task '{task_id}' not found. Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def get_task_by_difficulty(difficulty: str) -> Task:
    """Return the first task matching the given difficulty."""
    for task in TASKS.values():
        if task.observation.difficulty == difficulty:
            return task
    raise KeyError(f"No task found for difficulty '{difficulty}'")


def list_task_ids() -> list[str]:
    """Return all available task IDs."""
    return list(TASKS.keys())
