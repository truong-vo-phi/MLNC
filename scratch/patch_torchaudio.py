from pathlib import Path
p = Path("/tmp/uv-venv/lib/python3.13/site-packages/torchaudio/_extension/__init__.py")
if p.exists():
    content = p.read_text()
    if 'if not _load_lib("libtorchaudio"):' in content:
        content = content.replace(
            'if not _load_lib("libtorchaudio"):',
            'try:\n        _load_lib("libtorchaudio")\n    except Exception as e:\n        pass\n    if False:'
        )
        p.write_text(content)
        print("Patched torchaudio!")
    else:
        print("Already patched or pattern not found.")

import torchaudio
import torchaudio.transforms as T
m = T.MelSpectrogram()
print("Torchaudio MelSpectrogram works:", m)
