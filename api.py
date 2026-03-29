print("=== API FILE LOADED ===")

from __future__ import annotations
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from environment import CustomerSupportEnv
from models import Action

app = FastAPI(title="Customer Support OpenEnv")
env = CustomerSupportEnv()

@app.on_event("startup")
def startup():
    print("=== API STARTED SUCCESSFULLY ===")

@app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment. Accepts optional JSON: {"task_id": "..."}
    """
    try:
        body = {}
        try:
            raw = await request.body()
            if raw and raw.strip():
                body = json.loads(raw)
        except Exception:
            body = {}

        task_id = body.get("task_id", None)
        obs = env.reset(task_id=task_id)
        
        # OpenEnv expects observation at the top level
        return {"observation": obs.model_dump()}
    except Exception as e:
        print(f"Error in /reset: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/step")
async def step(request: Request):
    """
    Execute an action. Content: {"action": {"response": "..."}}
    """
    try:
        body = await request.json()
        action_data = body.get("action", body)
        
        action_obj = Action(**action_data)
        obs = env.step(action_obj)
        
        return {
            "observation": obs.model_dump(),
            "reward": float(obs.reward) if obs.reward is not None else 0.0,
            "done": bool(obs.done),
            "info": obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}
        }
    except Exception as e:
        print(f"Error in /step: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/state")
def get_state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Customer Support OpenEnv API is running"}
