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
import torch, json
print('training_history len:', len(training_history))
for h in training_history:
    m = h['metrics']
    print(f"Ep {h['epoch']}: train_loss={h['train_loss']:.4f}, val_top1={h['top1']:.2f}%, val_top3={m['top3']*100:.2f}%, map={m['map']*100:.2f}%, dur={h['duration_s']:.1f}s")

ckpt = torch.load('/marimo/checkpoints/best_model_enhanced.pt', map_location='cpu', weights_only=False)
print('CKPT EPOCH:', ckpt['epoch'])
print('CKPT METRICS:', json.dumps(ckpt['metrics'], indent=2))
print('CKPT SNR BREAKDOWN:', json.dumps(ckpt['snr_breakdown'], indent=2))
"""
print(run_code(code))
