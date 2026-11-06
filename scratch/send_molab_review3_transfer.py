import base64
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

source = Path('D:/PROJECT/MLNC/scratch/review3_molab_transfer.py').read_bytes()
chunks = [source[i:i + 3500] for i in range(0, len(source), 3500)]
def send(item):
    index, chunk = item
    target = f'/marimo/review-3/.ce_v2_part_{index:04d}'
    data = base64.b64encode(chunk).decode('ascii')
    code = f"import base64; f=open({target!r},'wb'); f.write(base64.b64decode({data!r})); f.close(); print('chunk {index + 1}/{len(chunks)}')"
    result = subprocess.run([
        r'C:\Program Files\Git\bin\bash.exe',
        'D:/PROJECT/MLNC/.agents/skills/marimo-pair/scripts/execute-code.sh',
        '--url', 'https://sb-a5e12fe063487262.sb.molab.run/',
        '--token', os.environ['MARIMO_PAIR_TOKEN'], '-c', code,
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Chunk {index + 1} failed: {result.stderr[-500:]} {result.stdout[-500:]}')
    return index + 1

with ThreadPoolExecutor(max_workers=5) as pool:
    for count, _ in enumerate(pool.map(send, enumerate(chunks)), 1):
        if count % 10 == 0 or count == len(chunks):
            print(f'Transferred {count}/{len(chunks)} chunks', flush=True)
