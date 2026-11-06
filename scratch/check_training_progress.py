import requests, json

URL = 'https://sb-83326b2d8a8049e4.sb.molab.run'
TOKEN = 'a6711c9c0fc6b1d77c8fdf2b6e9c985802f2e7c36d66984f54fd0382e1e612a1'
headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
s_resp = requests.get(f'{URL}/api/sessions', headers=headers)
headers['Marimo-Session-Id'] = list(s_resp.json().keys())[0]

def run_code(code):
    resp = requests.post(f'{URL}/api/kernel/execute', headers=headers, json={'code': code}, stream=True)
    out = []
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith('data: '):
            try:
                p = json.loads(line[6:])
                if 'data' in p and isinstance(p['data'], str):
                    out.append(p['data'])
            except: pass
    return ''.join(out)

code = """
from pathlib import Path
import json

p_sum = Path('/marimo/logs/summary_enhanced.json')
if p_sum.exists():
    d = json.loads(p_sum.read_text())
    print('TRAINING COMPLETED!')
    print('Best Epoch:', d.get('best_epoch'))
    print('Best Top-1:', d.get('best_top1'))
    print('Total epochs in history:', len(d.get('history', [])))
else:
    import torch
    ckpt = torch.load('/marimo/checkpoints/best_model_enhanced.pt', map_location='cpu', weights_only=False)
    print('TRAINING STILL RUNNING! Current best checkpoint at Epoch:', ckpt.get('epoch'))
    m = ckpt.get('metrics', {})
    print(f"Best Val Top-1: {m.get('top1', 0)*100:.2f}%, Top-3: {m.get('top3', 0)*100:.2f}%, mAP: {m.get('map', 0)*100:.2f}%")
"""
print(run_code(code))
