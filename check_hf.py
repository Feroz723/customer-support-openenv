"""Check if HF Space is live and test /reset endpoint."""
import urllib.request
import json
import time
import sys

BASE = "https://firoz369-customer-support-env.hf.space"

print("Checking HF Space...", flush=True)
for attempt in range(10):
    try:
        req = urllib.request.Request(f"{BASE}/health", method="GET")
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read().decode()
        print(f"[{attempt+1}] /health => {resp.status}: {data}", flush=True)
        if resp.status == 200:
            break
    except Exception as e:
        print(f"[{attempt+1}] Not ready yet: {type(e).__name__}", flush=True)
    time.sleep(15)
else:
    print("Space not ready after 10 attempts. Try again later.")
    sys.exit(1)

# Test /reset
print("\nTesting POST /reset...", flush=True)
req = urllib.request.Request(f"{BASE}/reset", data=b"{}", headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=15)
d = json.loads(resp.read())
print(f"Status: {resp.status}")
print(f"Top keys: {list(d.keys())}")
print(f"reward: {d.get('reward')}")
print(f"done: {d.get('done')}")
has_obs = "observation" in d
has_reward = "reward" in d
has_done = "done" in d
print(f"\n{'PASS' if has_obs and has_reward and has_done else 'FAIL'}: OpenEnv format check")
