from typing import Any

from fastapi import FastAPI
from environment import CustomerSupportEnv
from models import Action
import uvicorn

app = FastAPI()
env = CustomerSupportEnv()

@app.post("/reset")
def reset(body: dict[str, Any] | None = None):
    task_id = (body or {}).get("task_id")
    obs = env.reset(task_id=task_id)
    return {"observation": obs.model_dump()}

@app.post("/step")
def step(body: dict[str, Any]):
    # Map input schema to Action model
    action_data = body.get("action", body)
    action = Action(**action_data)
    obs = env.step(action)
    
    # Return OpenEnv compliant structure with top-level fields
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
        "done": bool(obs.done),
        "info": obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Customer Support OpenEnv API is running"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
