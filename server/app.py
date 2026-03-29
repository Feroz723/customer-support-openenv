from fastapi import FastAPI
from environment import CustomerSupportEnv
from models import Action

app = FastAPI()
env = CustomerSupportEnv()

@app.post("/reset")
def reset(body: dict = {}):
    task_id = body.get("task_id")
    obs = env.reset(task_id=task_id)
    return {"observation": obs.model_dump()}

@app.post("/step")
def step(body: dict):
    # Map input schema to Action model
    action_data = body.get("action", body)
    action = Action(**action_data)
    obs = env.step(action)
    
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
        "done": bool(obs.done),
        "info": obs.reward_breakdown.model_dump() if obs.reward_breakdown else {}
    }

@app.get("/state")
def get_state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok"}
