import marimo._code_mode as cm

code = '''mo.md("""### CE v2: 8 pretrained models

Code: `/marimo/review-3/ce_v2`. Checkpoints use the highest validation accuracy; ties keep the earliest epoch. The existing training cell still runs the original review-3 benchmark.

Extract the clean mixture-v2 dataset, then run from `/marimo/review-3/ce_v2`:

```bash
python configure_data.py --data-root /path/to/mix-dataset-v2
python prepare_ce9.py
python train_ce9_clean.py
```

For one model: `python train_ce9_clean.py --model ced_tiny`. Preparation downloads pretrained assets and checks all eight models. This update did not start training.
""")'''
async with cm.get_context() as ctx:
    cid = ctx.create_cell(code, after="training_config", name="ce_v2_guide", hide_code=True)
    ctx.run_cell(cid)
print("Created guide cell:", cid)
