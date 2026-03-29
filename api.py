print("=== LOADING API MODULE ===", flush=True)

"""
OpenEnv-compatible API wrapper for Customer Support Resolution Environment.
Exposes: POST /reset, POST /step, GET /state
"""
from __future__ import annotations

import json
import traceback

print("=== IMPORTS: fastapi ===", flush=True)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

print("=== IMPORTS: environment ===", flush=True)
from environment import CustomerSupportEnv

print("=== IMPORTS: models ===", flush=True)
from models import Action, RewardBreakdown

print("=== ALL IMPORTS OK ===", flush=True)

app = FastAPI(title="Customer Support OpenEnv")
env = CustomerSupportEnv()

print("=== ENV INITIALIZED ===", flush=True)


# ──────────────────────────────────────────────
# STARTUP EVENT
# ──────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    print("=== API SERVER STARTED ===", flush=True)
    print("=== READY TO ACCEPT REQUESTS ON PORT 7860 ===", flush=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def serialize_observation(obs) -> dict:
    """Convert Observation to OpenEnv wire format."""
    obs_dict = obs.model_dump()
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Request):
    """Reset environment. Accepts optional {"task_id": "..."}."""
    try:
        body = {}
        try:
            raw = await request.body()
            if raw and raw.strip():
                body = json.loads(raw)
        except Exception:
            body = {}

        task_id = body.get("task_id", None)
        difficulty = body.get("difficulty", None)

        print(f"[API] POST /reset task_id={task_id}", flush=True)

        obs = env.reset(task_id=task_id, difficulty=difficulty)
        response = serialize_observation(obs)

        print(f"[API] /reset OK", flush=True)
        return JSONResponse(content=response)

    except Exception as e:
        print(f"[API] /reset ERROR: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/step")
async def step(request: Request):
    """Execute action. Accepts {"action": {"response": "..."}}."""
    try:
        body = await request.json()
        print(f"[API] POST /step keys={list(body.keys())}", flush=True)

        action_data = body.get("action", body)
        action_obj = Action(**action_data)
        obs = env.step(action_obj)
        response = serialize_observation(obs)

        print(f"[API] /step OK reward={response['reward']}", flush=True)
        return JSONResponse(content=response)

    except Exception as e:
        print(f"[API] /step ERROR: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/state")
def get_state():
    """Return environment state."""
    return env.state()


@app.post("/state")
def post_state():
    """Return environment state (POST variant)."""
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "customer-support-env", "status": "running"}
