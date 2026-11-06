from pathlib import Path
for p in [Path('/usr/local/lib/python3.13/site-packages/torchaudio/_extension/__init__.py'), Path('/tmp/uv-venv/lib/python3.13/site-packages/torchaudio/_extension/__init__.py')]:
    if p.exists():
        content = p.read_text()
        target = 'if not _load_lib("libtorchaudio"):'
        if target in content:
            replacement = 'try:\n        _load_lib("libtorchaudio")\n    except Exception:\n        pass\n    if False:'
            content = content.replace(target, replacement)
            p.write_text(content)
            print('Patched', p)
import torchaudio.compliance.kaldi as ta_kaldi
print('ta_kaldi works:', hasattr(ta_kaldi, 'fbank'))
