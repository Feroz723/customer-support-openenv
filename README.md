---
title: Customer Support OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Customer Support Resolution Environment

A production-grade OpenEnv environment that simulates realistic customer support interactions for AI agent evaluation. Built for the NovaMart e-commerce platform.

## Overview

An AI agent receives a customer complaint (delayed order, refund request, damaged item, etc.) and must generate an appropriate support response. The environment **deterministically grades** the response on empathy, correctness, and helpfulness — producing a score from 0.0 to 1.0.

No LLM is used in the grading loop. Same input → same output, always.

---

## Tasks

| Task ID | Difficulty | Scenario |
|---|---|---|
| `easy_001` | Easy | Customer reports a delayed Bluetooth speaker order |
| `med_001` | Medium | Customer wants a refund for one item + expedited shipping for another |
| `hard_001` | Hard | Angry repeat customer — damaged item, missing refund, chargeback threat |

Each task includes realistic context: customer profile, order details, interaction history, internal notes, and relevant company policies.

---

## Observation (what the agent sees)

After calling `reset()`, the agent receives:

| Field | Type | Description |
|---|---|---|
| `ticket_id` | string | Support ticket ID |
| `user_query` | string | The customer's message |
| `sentiment` | string | `neutral`, `frustrated`, `angry`, or `anxious` |
| `category` | string | Issue category (e.g. `delayed_order`) |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `customer` | object | Name, tier, account age |
| `order` | object | Order ID, items, status, delivery date |
| `history` | object | Previous tickets, escalation count |
| `internal_notes` | list | Internal info visible to the agent |
| `company_policies` | dict | Relevant policy excerpts |

---

## Action (what the agent sends)

| Field | Type | Required | Description |
|---|---|---|---|
| `response` | string | Yes | The agent's text response to the customer (1–2000 chars) |
| `proposed_resolution` | string | No | Optional structured resolution code |
| `escalate` | bool | No | Whether to escalate the case |

---

## Scoring

Responses are graded across 4 dimensions:

| Dimension | Max Score | What it measures |
|---|---|---|
| **Empathy** | 0.30 | Apology, acknowledgement, personalisation, positive closing |
| **Correctness** | 0.40 | Addresses expected resolutions, follows company policy |
| **Helpfulness** | 0.30 | Clear next steps, timeline, follow-up info |
| **Penalty** | −0.20 | Off-topic, too short, forbidden/dismissive phrasing |

**Final score** = `empathy + correctness + helpfulness + penalty`, clamped to `[0.0, 1.0]`.

All scoring is keyword-based and fully deterministic.

---

## Quick Start

### Local

```bash
pip install -r requirements.txt

# Run smoke test
python environment.py

# Run inference (requires HuggingFace token)
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

### Docker

```bash
docker build -t support-env .

docker run \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_your_token_here" \
  support-env
```

---

## Project Structure

```
customer-support-env/
├── openenv.yaml         # OpenEnv manifest
├── environment.py       # CustomerSupportEnv class (reset/step/state)
├── models.py            # Pydantic data models (Observation, Action, RewardBreakdown)
├── tasks.py             # 3 task definitions with grading rubrics
├── grading.py           # Deterministic scoring engine
├── inference.py         # LLM inference script
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container setup
└── README.md            # This file
```

---

## API

```python
from environment import CustomerSupportEnv
from models import Action

env = CustomerSupportEnv()

# Start a new episode
obs = env.reset(task_id="easy_001")
print(obs.user_query)

# Agent responds
result = env.step(Action(response="Dear James, I sincerely apologize..."))
print(result.reward)             # 0.0 – 1.0
print(result.reward_breakdown)   # per-dimension scores

# Inspect state
state = env.state()

# Cleanup
env.close()
```

---

## License

MIT
