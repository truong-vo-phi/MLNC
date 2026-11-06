import os
import subprocess
import sys
from pathlib import Path

root = Path('/marimo/review-3/ce_v2')
code = r'''
import csv,json,numpy as np,soundfile as sf,torch
from pathlib import Path
from noise_pipeline.ce9_models import build_model
root=Path('/dataset/mix-dataset-v2')
with (root/'train/manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
    rows=[next(csv.DictReader(stream))]
row=rows[0]
wave,sr=sf.read(root/'train/mixture'/(row['sample_id']+'.wav'),dtype='float32')
cache=np.load('/dataset/cache/train.npy',mmap_mode='r')[0].copy()
assert sr == 16000 and wave.shape == cache.shape == (64000,)
label=int(np.argmax(json.loads(row['multi_hot_36'])))
model=build_model('panns_cnn10',Path('benchmark_results/ce9_clean_mixture_20261102/pretrained')).cuda()
optim=torch.optim.AdamW(model.parameters(),lr=3e-5)
for name,audio in [('mixture',wave),('frcrn',cache)]:
    model.train(); optim.zero_grad(set_to_none=True)
    x=torch.from_numpy(np.stack([audio,audio])).cuda()
    target=torch.tensor([label,label],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x).float(); loss=torch.nn.functional.cross_entropy(logits,target)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optim.step()
    print(name,'finite loss',float(loss),'optimizer step ok',flush=True)
'''
p = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, timeout=120)
print('exit:', p.returncode, '\n', p.stdout, p.stderr[-1800:])
if p.returncode:
    raise RuntimeError('Real-input optimizer smoke failed')
