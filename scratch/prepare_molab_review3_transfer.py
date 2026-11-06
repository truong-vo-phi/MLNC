from pathlib import Path
import base64
import io
import zipfile

root = Path('D:/PROJECT/MLNC/imp/review-3')
archive = io.BytesIO()
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in (root / 'ce_v2').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and 'benchmark_results' not in p.parts and p.suffix != '.pyc':
            z.write(p, p.relative_to(root).as_posix())
payload = base64.b64encode(archive.getvalue()).decode('ascii')
code = '''import base64, io, zipfile, pathlib, hashlib, datetime, shutil
root = pathlib.Path('/marimo/review-3')
dest = root / 'ce_v2'
if dest.exists():
    raise RuntimeError(f'Refusing to overwrite existing directory: {dest}')
trainer = root / 'src/trainer.py'
config = root / 'config/config.yaml'
t = trainer.read_text(encoding='utf-8')
c = config.read_text(encoding='utf-8')
old_default = 'self.config.get("logging", {}).get("metric_for_best", "macro_f1")'
old_select = '            current_metric = val_metrics.get(self.metric_for_best, val_metrics["macro_f1"])'
if t.count(old_default) != 1 or t.count(old_select) != 1 or c.count('  metric_for_best: macro_acc') != 1:
    raise RuntimeError('Remote trainer/config differ from inspected versions; no files changed')
archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD)))
for info in archive.infolist():
    path = pathlib.PurePosixPath(info.filename)
    if path.is_absolute() or '..' in path.parts or path.parts[0] != 'ce_v2':
        raise ValueError(f'Unsafe archive member: {info.filename}')
backup = root / ('backup_before_ce_v2_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir()
shutil.copy2(trainer, backup / 'trainer.py')
shutil.copy2(config, backup / 'config.yaml')
archive.extractall(root)
t = t.replace(old_default, 'self.config.get("logging", {}).get("metric_for_best", "acc")')
t = t.replace(old_select, '            if self.metric_for_best not in val_metrics:\\n                raise ValueError(f"Unknown checkpoint metric: {self.metric_for_best}")\\n            current_metric = val_metrics[self.metric_for_best]')
trainer.write_text(t, encoding='utf-8')
config.write_text(c.replace('  metric_for_best: macro_acc', '  metric_for_best: acc'), encoding='utf-8')
print('Installed files:', len(archive.infolist()))
print('Backup:', backup)
print('Archive SHA256:', hashlib.sha256(base64.b64decode(PAYLOAD)).hexdigest())
'''
code = code.replace('PAYLOAD', repr(payload))
out = Path('D:/PROJECT/MLNC/scratch/review3_molab_transfer.py')
out.write_text(code, encoding='utf-8')
print('Payload bytes:', out.stat().st_size)
