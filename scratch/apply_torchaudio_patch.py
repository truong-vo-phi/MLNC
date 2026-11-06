from pathlib import Path
p = Path('/usr/local/lib/python3.13/site-packages/torchaudio/__init__.py')
content = p.read_text()
target = 'from . import compliance, datasets, functional, models, pipelines, transforms, utils  # noqa: F401'
replacement = 'from . import compliance\ntry:\n    from . import datasets, functional, models, pipelines, transforms, utils\nexcept Exception:\n    pass'
if target in content:
    content = content.replace(target, replacement)
    p.write_text(content)
    print('Successfully patched torchaudio __init__.py')

import torchaudio.compliance.kaldi as ta_kaldi
print('torchaudio.compliance.kaldi successfully imported!', hasattr(ta_kaldi, 'fbank'))
