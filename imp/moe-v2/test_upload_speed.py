import sys
import time
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import marimo_client

# Generate 5MB of random bytes
test_data = b"0" * (5 * 1024 * 1024)
b64_data = base64.b64encode(test_data).decode("ascii")

code = f"""
import base64
data = base64.b64decode('{b64_data}')
print('Received chunk on remote, len:', len(data))
"""

t0 = time.time()
marimo_client.execute_remote(code)
t1 = time.time()
print(f"Uploaded 5MB in {t1 - t0:.2f}s -> Rate: {5.0 / (t1 - t0):.2f} MB/s")
