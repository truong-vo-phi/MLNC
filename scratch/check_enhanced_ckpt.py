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
import torch
ckpt = torch.load('/marimo/checkpoints/best_model_enhanced.pt', map_location='cpu', weights_only=False)
print('Enhanced Checkpoint keys:', list(ckpt.keys()))
print('Epoch:', ckpt.get('epoch'))
m = ckpt.get('metrics', {})
print(f"Top-1: {m.get('top1', 0)*100:.2f}%, Top-3: {m.get('top3', 0)*100:.2f}%, mAP: {m.get('map', 0)*100:.2f}%, AUC: {m.get('auc', 0)*100:.2f}%")
print('SNR Breakdown:')
for k, v in ckpt.get('snr_breakdown', {}).items():
    if isinstance(v, dict):
        print(f"  {k}: Top-1={v.get('top1', 0)*100:.2f}%, Alpha={v.get('mean_alpha', 0.5):.3f}")
"""
print(run_code(code))
