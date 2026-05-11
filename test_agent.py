import subprocess
import time
import urllib.request
import json

p = subprocess.Popen(["../.venv/bin/python", "-m", "uvicorn", "healthcare_agent.app:a2a_app", "--port", "8001"], cwd="po-adk-python")
for _ in range(10):
    time.sleep(1)
    try:
        req = urllib.request.Request("http://127.0.0.1:8001/.well-known/agent-card.json")
        with urllib.request.urlopen(req) as response:
            print(json.dumps(json.loads(response.read()), indent=2))
            break
    except Exception:
        pass
finally:
    p.terminate()
