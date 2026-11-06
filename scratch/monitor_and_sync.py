import requests, json, time

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

for i in range(15):
    code = """
import os
from pathlib import Path
p_sum = Path('/marimo/logs/summary_enhanced.json')
if p_sum.exists():
    print('FINISHED')
else:
    print(f'EPOCH={epoch}, PATIENCE={patience_counter}/8, BEST={best_epoch}, HISTORY_LEN={len(training_history)}')
"""
    res = run_code(code).strip()
    print(f"[{time.strftime('%H:%M:%S')}] {res}")
    if 'FINISHED' in res:
        break
    time.sleep(15)
