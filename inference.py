from __future__ import annotations
"""
Inference script for Customer Support Resolution Environment.
Connects the OpenEnv environment to an LLM via OpenAI-compatible API.
Runs all 3 tasks, collects scores, prints results.

Usage:
    set API_BASE_URL=https://api-inference.huggingface.co/v1
    set MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    set HF_TOKEN=hf_...
    python inference.py
"""
print("=== CONTAINER STARTED ===", flush=True)
import os
import sys
import json
import time

from openai import OpenAI

from environment import CustomerSupportEnv
from models import Action
from tasks import list_task_ids

print("=== IMPORTS SUCCESS ===", flush=True)

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════

# Debug prints to verify environment variables inside the container
print("DEBUG ENV:", flush=True)
api_env = os.getenv("API_BASE_URL")
print(f"  API_BASE_URL (env): {api_env}", flush=True)
print(f"  MODEL_NAME:   {os.getenv('MODEL_NAME')}", flush=True)
print(f"  HF_TOKEN exists: {bool(os.getenv('HF_TOKEN'))}", flush=True)

# Use env vars with fallbacks (Updated to new HF router endpoint)
API_BASE_URL = api_env or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

TEMPERATURE = 0.2       # low for reproducibility
MAX_TOKENS = 512        # enough for a full support response


# ══════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════

def build_prompt(obs) -> list[dict]:
    """
    Build chat messages from an Observation.
    System prompt sets the agent persona.
    User message contains the customer query + context.
    """
    # Build context string from available data
    context_parts = []

    if obs.customer:
        context_parts.append(
            f"Customer: {obs.customer.name} | "
            f"Tier: {obs.customer.tier} | "
            f"Account age: {obs.customer.account_age_months} months"
        )

    if obs.order:
        context_parts.append(
            f"Order: {obs.order.order_id} | "
            f"Items: {', '.join(obs.order.items)} | "
            f"Status: {obs.order.status} | "
            f"Expected delivery: {obs.order.expected_delivery}"
        )
        if obs.order.tracking_number:
            context_parts.append(f"Tracking: {obs.order.tracking_number}")

    if obs.history:
        context_parts.append(
            f"Previous tickets: {obs.history.previous_tickets} | "
            f"Escalations: {obs.history.escalation_count}"
        )

    if obs.internal_notes:
        context_parts.append("Internal notes:\n" + "\n".join(f"  - {n}" for n in obs.internal_notes))

    if obs.company_policies:
        policy_text = "\n".join(f"  - {k}: {v}" for k, v in obs.company_policies.items())
        context_parts.append(f"Company policies:\n{policy_text}")

    context_block = "\n".join(context_parts)

    system_message = (
        "You are a professional customer support agent for NovaMart, an online retailer. "
        "Respond to the customer's query with empathy, accuracy, and clear next steps.\n\n"
        "Guidelines:\n"
        "- Always apologize sincerely for any inconvenience\n"
        "- Acknowledge the customer's frustration or concern\n"
        "- Address the customer by name\n"
        "- Provide a clear, correct solution based on company policies\n"
        "- Include specific timelines and next steps\n"
        "- Offer follow-up contact information\n"
        "- End with a positive, helpful closing\n"
        "- Be concise but thorough"
    )

    user_message = (
        f"=== CUSTOMER SUPPORT TICKET ===\n"
        f"Ticket ID: {obs.ticket_id}\n"
        f"Sentiment: {obs.sentiment}\n"
        f"Category: {obs.category}\n\n"
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== CUSTOMER MESSAGE ===\n{obs.user_query}\n\n"
        f"Please write your response to the customer."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ══════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════

def call_llm(client: OpenAI, messages: list[dict]) -> str:
    """Call the LLM and return the response text."""
    try:
        print("=== BEFORE LLM CALL ===", flush=True)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        print("=== LLM CALL SUCCESS ===", flush=True)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_msg += f" | Body: {e.response.text}"
        print(f"  ⚠ LLM API error: {type(e).__name__}: {error_msg}", flush=True)
        return ""


# ══════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ══════════════════════════════════════════════

def main():
    try:
        # ── Validate config ──
        if not HF_TOKEN:
            raise ValueError("❌ ERROR: HF_TOKEN is missing or empty. Please set it in HF Space Secrets.")

        print("=" * 60, flush=True)
        print("CUSTOMER SUPPORT RESOLUTION — INFERENCE", flush=True)
        print("=" * 60, flush=True)
        print(f"  API Base: {API_BASE_URL}", flush=True)
        print(f"  Model:    {MODEL_NAME}", flush=True)
        print(f"  Token:    {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}", flush=True)
        print(flush=True)

        # ── Init client ──
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

        # ── Init environment ──
        env = CustomerSupportEnv()
        task_ids = list_task_ids()

        results = []

        # ── Run each task ──
        for task_id in task_ids:
            print(f"{'─' * 60}", flush=True)
            print(f"▶ Task: {task_id}", flush=True)

            # 1. Reset environment
            obs = env.reset(task_id=task_id)
            print(f"  Difficulty: {obs.difficulty}", flush=True)
            print(f"  Customer:   {obs.customer.name} ({obs.customer.tier})", flush=True)
            print(f"  Sentiment:  {obs.sentiment}", flush=True)
            print(f"  Query:      {obs.user_query[:80]}...", flush=True)

            # 2. Build prompt
            messages = build_prompt(obs)

            # 3. Call LLM
            print(f"  Calling LLM...", flush=True)
            response_text = call_llm(client, messages)

            if not response_text:
                print("  ⚠ Empty response from LLM — skipping.", flush=True)
                results.append({"task_id": task_id, "score": 0.0, "error": "empty_response"})
                continue

            print(f"  Response:   {response_text[:100]}...", flush=True)
            print(f"  Length:     {len(response_text)} chars", flush=True)

            # 4. Step environment with agent response
            action = Action(response=response_text)
            obs_result = env.step(action)

            # 5. Collect reward
            breakdown = obs_result.reward_breakdown
            score = obs_result.reward

            print(f"\n  ── SCORES ──", flush=True)
            print(f"  Empathy:      {breakdown.empathy_score:.2f} / 0.30", flush=True)
            print(f"  Correctness:  {breakdown.correctness_score:.2f} / 0.40", flush=True)
            print(f"  Helpfulness:  {breakdown.helpfulness_score:.2f} / 0.30", flush=True)
            print(f"  Penalty:      {breakdown.penalty:.2f}", flush=True)
            print(f"  TOTAL:        {score:.2f} / 1.00", flush=True)

            results.append({
                "task_id": task_id,
                "difficulty": obs.difficulty,
                "score": score,
                "empathy": breakdown.empathy_score,
                "correctness": breakdown.correctness_score,
                "helpfulness": breakdown.helpfulness_score,
                "penalty": breakdown.penalty,
                "response_length": len(response_text),
            })

        # ── Summary ──
        print(f"\n{'=' * 60}", flush=True)
        print("RESULTS SUMMARY", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"{'Task':<12} {'Difficulty':<10} {'Score':<8}", flush=True)
        print(f"{'─' * 30}", flush=True)

        total_score = 0.0
        for r in results:
            score = r.get("score", 0.0)
            diff = r.get("difficulty", "?")
            print(f"{r['task_id']:<12} {diff:<10} {score:.2f}", flush=True)
            total_score += score

        avg_score = total_score / len(results) if results else 0.0
        print(f"{'─' * 30}", flush=True)
        print(f"{'AVERAGE':<22} {avg_score:.2f}", flush=True)
        print(flush=True)

        # ── Save results to JSON ──
        output_file = "inference_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "results": results,
                "average_score": round(avg_score, 4),
            }, f, indent=2)
        print(f"📄 Results saved to {output_file}", flush=True)

        # ── Cleanup ──
        env.close()
        print("✅ Inference complete.", flush=True)
        
        # Keep container alive briefly so logs are captured
        time.sleep(5)

        return avg_score
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
