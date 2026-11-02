with open('/marimo/moe-v2/src/frontend.py', 'r') as f:
    c = f.read()
if 'TARGET_RMS = 0.07925' not in c:
    c = c.replace('RMS_EPS = 1e-8', 'RMS_EPS = 1e-8\nTARGET_RMS = 0.07925')
    with open('/marimo/moe-v2/src/frontend.py', 'w') as f:
        f.write(c)

import sys
for k in list(sys.modules.keys()):
    if k == 'src' or k.startswith('src.'):
        del sys.modules[k]

if '/marimo/moe-v2' in sys.path:
    sys.path.remove('/marimo/moe-v2')

print('Cleaned sys.modules and ensured TARGET_RMS in moe-v2 frontend!')
