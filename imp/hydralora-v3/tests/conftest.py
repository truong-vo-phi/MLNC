import sys
from pathlib import Path

HYDRALORA_DIR = Path(__file__).resolve().parent.parent
for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]
if str(HYDRALORA_DIR) not in sys.path:
    sys.path.insert(0, str(HYDRALORA_DIR))
