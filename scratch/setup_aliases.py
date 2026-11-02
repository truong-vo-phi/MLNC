import os
import sys

# 1. In /marimo/moe-v2/src/models, create symlinks/aliases so both naming conventions work
models_dir = "/marimo/moe-v2/src/models"
if os.path.exists(models_dir):
    vg = os.path.join(models_dir, "vit_grep.py")
    vb = os.path.join(models_dir, "vit_bias.py")
    if not os.path.exists(vg) and os.path.exists(vb):
        try:
            os.symlink(vb, vg)
            print("Created symlink vit_grep.py -> vit_bias.py")
        except:
            import shutil
            shutil.copy(vb, vg)
            print("Copied vit_grep.py from vit_bias.py")

    sm = os.path.join(models_dir, "split_moe.py")
    ms = os.path.join(models_dir, "moe_split.py")
    if not os.path.exists(sm) and os.path.exists(ms):
        try:
            os.symlink(ms, sm)
            print("Created symlink split_moe.py -> moe_split.py")
        except:
            import shutil
            shutil.copy(ms, sm)
            print("Copied split_moe.py from moe_split.py")

# 2. Clean sys.modules
for k in list(sys.modules.keys()):
    if k == "src" or k.startswith("src."):
        del sys.modules[k]

print("Cleaned sys.modules!")
