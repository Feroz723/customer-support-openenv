from __future__ import annotations
from fastapi import FastAPI, Body
from pydantic import BaseModel
from environment import CustomerSupportEnv
from models import Action
import uvicorn

app = FastAPI(title="Customer Support OpenEnv API")
env = CustomerSupportEnv()

# ──────────────────────────────────────────────
# REQUEST MODELS
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str | None = None

class StepRequest(BaseModel):
    action: dict

# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment to a new episode.
    Expected Input: {"task_id": "..."} (optional)
    """
    obs = env.reset(task_id=req.task_id)
    
    # Return strict JSON structure
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs
    }

@app.post("/step")
def step(req: StepRequest):
    """
    Execute an action and return the result.
    Expected Input: {"action": {"response": "..."}}
    """
    # Map raw dict action to Action model
    action_obj = Action(**req.action)
    obs = env.step(action_obj)
    
    # Return strict OpenEnv response structure
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
        "done": bool(obs.done),
        "info": {
            "reward_breakdown": obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}
        }
    }

@app.get("/state")
def state():
    """Return the current internal state of the environment."""
    return env.state()

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "env": "customer-support-env"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
