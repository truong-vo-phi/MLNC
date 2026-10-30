import requests

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"
headers = {"Authorization": f"Bearer {TOKEN}"}

r1 = requests.get(f"{URL}/api/status", headers=headers, timeout=5)
print("status:", r1.status_code, r1.text)

r2 = requests.get(f"{URL}/api/sessions", headers=headers, timeout=5)
print("sessions:", r2.status_code, r2.text)
