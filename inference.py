from __future__ import annotations

import json
import os
import re
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


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(value: str | None) -> str:
    return "null" if value is None else _normalize_text(value)


def _format_reward(value: float) -> str:
    return f"{max(0.01, min(value, 0.99)):.2f}"


def _format_rewards(values: list[float]) -> str:
    return ",".join(_format_reward(value) for value in values)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={_normalize_text(action)} "
        f"reward={_format_reward(reward)} done={_format_bool(done)} error={_format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    print(
        f"[END] success={_format_bool(success)} steps={steps} rewards={_format_rewards(rewards)}",
        flush=True,
    )


def _fallback_action(observation: Observation) -> str:
    return _normalize_text(
        f"Hello {observation.customer.name}, I am sorry for the trouble. "
        "I am reviewing your request carefully and taking the next support step now."
    )


def build_messages(observation: Observation) -> list[dict[str, str]]:
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


def get_model_message(client: OpenAI, observation: Observation) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_messages(observation),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    message = (completion.choices[0].message.content or "").strip()
    return _normalize_text(message[:2000])


def run_task(client: OpenAI, task_id: str) -> dict[str, Any]:
    env = CustomerSupportEnv()
    rewards: list[float] = []
    steps_taken = 0
    success = False
    observation: Observation | None = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            error: str | None = None
            try:
                action_text = get_model_message(client, observation)
            except Exception as exc:
                error = str(exc)
                action_text = _fallback_action(observation)

            try:
                result = env.step(Action(response=action_text))
            except Exception as exc:
                error = str(exc) if error is None else error
                log_step(step=step, action=action_text, reward=0.01, done=False, error=error)
                break

            reward = float(result.reward or 0.01)
            done = bool(result.done)
            rewards.append(max(0.01, min(reward, 0.99)))
            steps_taken = step

            log_step(step=step, action=action_text, reward=reward, done=done, error=error)
            observation = result

            if done:
                break

        if rewards:
            success = (sum(rewards) / len(rewards)) >= SUCCESS_SCORE_THRESHOLD

        breakdown = (
            observation.reward_breakdown.model_dump() if observation and observation.reward_breakdown else {}
        )
        return {
            "task_id": task_id,
            "difficulty": observation.difficulty if observation else None,
            "score": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "success": success,
            "rewards": [round(reward, 4) for reward in rewards],
            "reward_breakdown": breakdown,
        }
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results: list[dict[str, Any]] = []

    for task_id in list_task_ids():
        results.append(run_task(client, task_id))

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
