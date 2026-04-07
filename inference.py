from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

from environment import CustomerSupportEnv
from models import Action, Observation
from tasks import list_task_ids

BENCHMARK = "customer-support-env"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TEMPERATURE = 0.2
MAX_TOKENS = 512
MAX_STEPS = 1
SUCCESS_SCORE_THRESHOLD = 0.5


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _clamp_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_json(task)} env={_json(env)} model={_json(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        "[STEP] "
        f"step={step} "
        f"action={_json(action)} "
        f"reward={reward:.4f} "
        f"done={_json(done)} "
        f"error={_json(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    normalized_rewards = [round(_clamp_score(reward), 4) for reward in rewards]
    print(
        "[END] "
        f"success={_json(success)} "
        f"steps={steps} "
        f"score={round(_clamp_score(score), 4):.4f} "
        f"rewards={_json(normalized_rewards)}",
        flush=True,
    )


def _fallback_response(observation: Observation) -> str:
    customer_name = observation.customer.name if observation.customer else "there"
    return (
        f"Hi {customer_name}, I am sorry for the trouble you have experienced. "
        "I am reviewing your case now, confirming the relevant order details, and taking the "
        "next appropriate support action based on the company policy. I will follow up with the "
        "resolution steps and timeline right away."
    )


def build_messages(
    observation: Observation,
    history: list[str],
    last_reward: float,
) -> list[dict[str, str]]:
    context_parts: list[str] = [
        f"Ticket ID: {observation.ticket_id}",
        f"Customer: {observation.customer.name} ({observation.customer.tier})",
        f"Sentiment: {observation.sentiment}",
        f"Category: {observation.category}",
        f"Difficulty: {observation.difficulty}",
    ]

    if observation.order:
        context_parts.extend(
            [
                f"Order ID: {observation.order.order_id}",
                f"Order items: {', '.join(observation.order.items)}",
                f"Order status: {observation.order.status}",
                f"Expected delivery: {observation.order.expected_delivery}",
            ]
        )
        if observation.order.tracking_number:
            context_parts.append(f"Tracking number: {observation.order.tracking_number}")

    if observation.internal_notes:
        context_parts.append("Internal notes:")
        context_parts.extend(f"- {note}" for note in observation.internal_notes)

    if observation.company_policies:
        context_parts.append("Company policies:")
        context_parts.extend(
            f"- {policy}: {details}"
            for policy, details in observation.company_policies.items()
        )

    if history:
        context_parts.append("Previous rollout history:")
        context_parts.extend(f"- {entry}" for entry in history)
        context_parts.append(f"Last reward: {last_reward:.4f}")

    system_prompt = (
        "You are a professional NovaMart customer support agent. "
        "Write a concise response that resolves the customer's problem correctly. "
        "Prioritize policy compliance, concrete next steps, accurate timelines, and empathy. "
        "If security recovery is needed, handle that before refunds or billing actions."
    )

    user_prompt = (
        "Use the context below to answer the customer.\n\n"
        f"{os.linesep.join(context_parts)}\n\n"
        f"Customer message:\n{observation.user_query}\n\n"
        "Respond as the support agent."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_model_message(
    client: OpenAI | None,
    observation: Observation,
    history: list[str],
    last_reward: float,
) -> tuple[str, str | None]:
    if client is None:
        return _fallback_response(observation), "missing_api_key"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_messages(observation, history, last_reward),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        message = (completion.choices[0].message.content or "").strip()
        if not message:
            return _fallback_response(observation), "empty_model_response"
        return message[:2000], None
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return _fallback_response(observation), type(exc).__name__


def run_task(client: OpenAI | None, env: CustomerSupportEnv, task_id: str) -> dict[str, Any]:
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    observation = env.reset(task_id=task_id)
    last_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if observation.done:
            break

        action_text, error = get_model_message(client, observation, history, last_reward)
        result = env.step(Action(response=action_text))

        reward = _clamp_score(float(result.reward or 0.0))
        done = bool(result.done)

        rewards.append(reward)
        steps_taken = step
        last_reward = reward

        log_step(step=step, action=action_text, reward=reward, done=done, error=error)
        history.append(f"Step {step}: reward={reward:.4f}")

        observation = result
        if done:
            break

    if rewards:
        score = _clamp_score(sum(rewards) / MAX_STEPS)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    breakdown = observation.reward_breakdown.model_dump() if observation.reward_breakdown else {}
    return {
        "task_id": task_id,
        "difficulty": observation.difficulty,
        "score": round(score, 4),
        "success": success,
        "rewards": [round(reward, 4) for reward in rewards],
        "reward_breakdown": breakdown,
    }


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = CustomerSupportEnv()
    results: list[dict[str, Any]] = []

    try:
        for task_id in list_task_ids():
            results.append(run_task(client, env, task_id))
    finally:
        env.close()

    average_score = round(
        sum(result["score"] for result in results) / len(results) if results else 0.0,
        4,
    )
    payload = {
        "env": BENCHMARK,
        "model": MODEL_NAME,
        "average_score": average_score,
        "results": results,
    }
    with open("inference_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
