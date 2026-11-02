import urllib.request
import json

url = "https://sb-c04297299c79cc0d.sb.molab.run/api/kernel/cells"
req = urllib.request.Request(
    url,
    headers={"Marimo-Server-Token": "95900018fcf7379c970f9596ce7758087d253b3c7d0cca8040f51f30edc0729e"}
)

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"Total cells: {len(data)}")
        for i, c in enumerate(data):
            cell_id = c.get("cell_id", "")
            name = c.get("name", "")
            code_lines = c.get("code", "").split("\n")
            preview = code_lines[0] if code_lines else ""
            print(f"Cell {i:02d} [{cell_id}] ({name}): {preview[:70]} ({len(code_lines)} lines)")
except Exception as e:
    print("Error:", e)
