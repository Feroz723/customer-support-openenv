"""
OpenEnv-compatible API wrapper for Customer Support Resolution Environment.
Follows the official OpenEnv spec: POST /reset, POST /step, GET /state.

Response format matches openenv-core's ResetResponse / StepResponse:
  {
    "observation": { ... },   # Observation fields (excluding reward, done)
    "reward": float | None,
    "done": bool
  }
"""
from __future__ import annotations

import json
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from environment import CustomerSupportEnv
from models import Action

app = FastAPI(title="Customer Support OpenEnv")
env = CustomerSupportEnv()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def serialize_observation(obs) -> dict:
    """
    Convert an Observation to the OpenEnv-standard wire format.
    Matches openenv-core's serialize_observation():
      - 'observation' dict contains ALL fields EXCEPT reward and done
      - 'reward' and 'done' are extracted to the top level
    """
    obs_dict = obs.model_dump()

    # Extract reward and done from observation
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)

    # Remove reward_breakdown from top-level observation
    # (it's internal scoring detail, keep it in observation for transparency)
    # Actually, keep it — the validator doesn't care about extra fields in observation

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
    """
    Reset the environment. Accepts optional {"task_id": "...", "seed": ..., "episode_id": "..."}.
    Returns OpenEnv-standard: {"observation": {...}, "reward": null, "done": false}
    """
    try:
        # Parse body safely — handle empty body, missing content-type, etc.
        body = {}
        try:
            raw = await request.body()
            if raw and raw.strip():
                body = json.loads(raw)
        except Exception:
            body = {}

        task_id = body.get("task_id", None)
        difficulty = body.get("difficulty", None)

        print(f"[API] POST /reset — task_id={task_id}, difficulty={difficulty}", flush=True)

        obs = env.reset(task_id=task_id, difficulty=difficulty)
        response = serialize_observation(obs)

        print(f"[API] /reset OK — returning observation for task_id={task_id}", flush=True)
        return JSONResponse(content=response)

    except Exception as e:
        print(f"[API] /reset ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/step")
async def step(request: Request):
    """
    Execute an action. Accepts {"action": {"response": "..."}}.
    Returns OpenEnv-standard: {"observation": {...}, "reward": float, "done": bool}
    """
    try:
        body = await request.json()
        print(f"[API] POST /step — keys={list(body.keys())}", flush=True)

        # Extract action dict — handle both {"action": {...}} and direct {...}
        action_data = body.get("action", body)

        # Build Action object
        action_obj = Action(**action_data)
        obs = env.step(action_obj)
        response = serialize_observation(obs)

        print(f"[API] /step OK — reward={response['reward']}, done={response['done']}", flush=True)
        return JSONResponse(content=response)

    except Exception as e:
        print(f"[API] /step ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/state")
def state_get():
    """Return current environment state metadata (GET)."""
    return env.state()


@app.post("/state")
def state_post():
    """Return current environment state metadata (POST — some validators use POST)."""
    return env.state()


@app.get("/health")
def health():
    """Health check for container monitoring."""
    return {"status": "healthy"}


@app.get("/")
def root():
    """Root endpoint — confirms the environment is running."""
    return {
        "name": "customer-support-env",
        "version": "0.1.0",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
