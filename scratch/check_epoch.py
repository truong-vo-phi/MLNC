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
print(f'Current loop epoch: {epoch}, patience_counter: {patience_counter}, best_epoch: {best_epoch}')
if 'training_history' in globals() and training_history:
    print(f'Total epochs recorded: {len(training_history)}')
    latest = training_history[-1]
    print(f'Latest epoch: {latest.get("epoch")}, train_loss: {latest.get("train_loss"):.4f}, top1: {latest.get("top1"):.2f}%')
    print('All epochs top1:', [round(h['top1'], 2) for h in training_history])
"""
print(run_code(code))
