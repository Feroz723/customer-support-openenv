"""Quick test of the local API server."""
import json
import time
import urllib.request

BASE = "http://localhost:7860"

# Test /reset
print("=== POST /reset ===")
req = urllib.request.Request(f"{BASE}/reset", data=b"{}", headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
print(f"Status: {resp.status}")
print(f"Top keys: {list(data.keys())}")
print(f"reward: {data.get('reward')}")
print(f"done: {data.get('done')}")
print(f"obs keys: {list(data.get('observation', {}).keys())}")

time.sleep(0.5)

# Test /step
print("\n=== POST /step ===")
step_body = json.dumps({"action": {"response": "Hello customer, I apologize for the delay."}}).encode()
req2 = urllib.request.Request(f"{BASE}/step", data=step_body, headers={"Content-Type": "application/json"})
resp2 = urllib.request.urlopen(req2)
data2 = json.loads(resp2.read())
print(f"Status: {resp2.status}")
print(f"Top keys: {list(data2.keys())}")
print(f"reward: {data2.get('reward')}")
print(f"done: {data2.get('done')}")

time.sleep(0.5)

# Test /state
print("\n=== GET /state ===")
resp3 = urllib.request.urlopen(f"{BASE}/state")
data3 = json.loads(resp3.read())
print(f"Status: {resp3.status}")
print(f"Keys: {list(data3.keys())}")

print("\n✅ All endpoints working!")
