from __future__ import annotations
"""
Inference script for Customer Support Resolution Environment.
Connects the OpenEnv environment to an LLM via OpenAI-compatible API.
Runs all tasks, collects scores, and prints audit-ready results.
"""

import os
import sys
import json
import time
from openai import OpenAI

from environment import CustomerSupportEnv
from models import Action
from tasks import list_task_ids

# Use env vars with fallbacks (Updated to new HF router endpoint)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

TEMPERATURE = 0.2
MAX_TOKENS = 512

def build_prompt(obs) -> list[dict]:
    """Build chat messages from an Observation."""
    context_parts = []

    if obs.customer:
        context_parts.append(
            f"Customer: {obs.customer.name} | Tier: {obs.customer.tier} | "
            f"Account age: {obs.customer.account_age_months} months"
        )

    if obs.order:
        context_parts.append(
            f"Order: {obs.order.order_id} | Status: {obs.order.status} | "
            f"Expected delivery: {obs.order.expected_delivery}"
        )

    if obs.internal_notes:
        context_parts.append("Internal notes:\n" + "\n".join(f"  - {n}" for n in obs.internal_notes))

    if obs.company_policies:
        policy_text = "\n".join(f"  - {k}: {v}" for k, v in obs.company_policies.items())
        context_parts.append(f"Company policies:\n{policy_text}")

    context_block = "\n".join(context_parts)

    system_prompt = (
        "You are a professional customer support agent for NovaMart. "
        "Respond with empathy, accuracy, and clear next steps.\n"
        "Guidelines:\n"
        "- Apologize sincerely; Acknowledge frustration; Address customer by name;\n"
        "- Provide correct solutions based on policies; Include specific timelines;\n"
        "- Offer follow-up contact and end with a helpful closing."
    )

    user_message = (
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== CUSTOMER MESSAGE ===\n{obs.user_query}\n\n"
        "Please write your response to the customer."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

def call_llm(client: OpenAI, messages: list[dict]) -> str:
    """Call the LLM and return the response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ LLM API error: {type(e).__name__}: {e}", flush=True)
        return ""

def main():
    try:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is missing. Set it in Environment Secrets.")

        print("=" * 60, flush=True)
        print("CUSTOMER SUPPORT RESOLUTION — INFERENCE", flush=True)
        print("=" * 60, flush=True)
        print(f"  Model: {MODEL_NAME}", flush=True)
        print(flush=True)

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        env = CustomerSupportEnv()
        task_ids = list_task_ids()
        results = []

        for task_id in task_ids:
            print(f"{'─' * 60}", flush=True)
            print(f"▶ Task: {task_id}", flush=True)

            obs = env.reset(task_id=task_id)
            print(f"  Difficulty: {obs.difficulty}", flush=True)
            print(f"  Customer:   {obs.customer.name}", flush=True)
            print(f"  Query:      {obs.user_query[:80]}...", flush=True)

            messages = build_prompt(obs)
            response_text = call_llm(client, messages)

            if not response_text:
                print("  ⚠ Skipping task due to error.", flush=True)
                continue

            # Enforcement: Truncate to 2000 chars before Action
            if len(response_text) > 2000:
                response_text = response_text[:2000]

            action = Action(response=response_text)
            obs_result = env.step(action)
            breakdown = obs_result.reward_breakdown
            score = obs_result.reward

            print(f"\n  ── SCORES ──", flush=True)
            print(f"  Empathy:      {breakdown.empathy_score:.2f} / 0.30", flush=True)
            print(f"  Correctness:  {breakdown.correctness_score:.2f} / 0.50", flush=True)
            print(f"  Helpfulness:  {breakdown.helpfulness_score:.2f} / 0.20", flush=True)
            print(f"  Penalty:      {breakdown.penalty:.2f}", flush=True)
            print(f"  TOTAL:        {score:.2f} / 1.00", flush=True)

            print(f"\n  ── REASONING ──", flush=True)
            for dim, info in breakdown.reasoning.items():
                details = info.get('details') if isinstance(info, dict) else info
                print(f"  {dim.capitalize():<12}: {details}", flush=True)

            results.append({
                "task_id": task_id,
                "difficulty": obs.difficulty,
                "score": score,
                "reasoning": breakdown.reasoning,
            })

        print(f"\n{'=' * 60}", flush=True)
        print("RESULTS SUMMARY", flush=True)
        print(f"{'=' * 60}", flush=True)
        for r in results:
            print(f"{r['task_id']:<12} {r['difficulty']:<10} {r['score']:.2f}", flush=True)

        avg_score = sum(r['score'] for r in results) / len(results) if results else 0.0
        print(f"{'─' * 30}", flush=True)
        print(f"{'AVERAGE':<22} {avg_score:.2f}", flush=True)

        with open("inference_results.json", "w") as f:
            json.dump({"results": results, "average_score": round(avg_score, 4)}, f, indent=2)

        env.close()
        print("\n✅ Inference complete.", flush=True)
        time.sleep(2)
        return avg_score

    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
