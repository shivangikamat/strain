import subprocess
import json

p = subprocess.Popen([".venv/bin/python", "-m", "mcp_server.server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
req = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
}
p.stdin.write(json.dumps(req) + "\n")
p.stdin.flush()
response = p.stdout.readline()
print(response)
p.terminate()
