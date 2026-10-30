import zipfile
from pathlib import Path

out_zip = Path(r"D:\PROJECT\MLNC\manifests.zip")

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    # train manifest
    train_m = Path(r"D:\content\mix-dataset-v2\train\manifest.csv")
    if train_m.is_file():
        zf.write(train_m, "train/manifest.csv")
    # validation manifest from eval (3064 clips, matches validation.npy)
    eval_m = Path(r"D:\content\mix-dataset-v2\eval\manifest.csv")
    if eval_m.is_file():
        zf.write(eval_m, "validation/manifest.csv")
        zf.write(eval_m, "eval/manifest.csv")
    # full 6480 validation manifest as all_validation/manifest.csv
    full_val_m = Path(r"D:\content\mix-dataset\validation\manifest.csv")
    if full_val_m.is_file():
        zf.write(full_val_m, "all_validation/manifest.csv")
    # test manifest
    test_m = Path(r"D:\content\mix-dataset-v2\test\manifest.csv")
    if test_m.is_file():
        zf.write(test_m, "test/manifest.csv")
    # labels
    labels_file = Path(r"D:\content\mix-dataset-v2\labels.txt")
    if labels_file.is_file():
        zf.write(labels_file, "labels.txt")
    # validation_clean
    val_clean = Path(r"D:\content\mix-dataset\validation_clean.csv")
    if val_clean.is_file():
        zf.write(val_clean, "validation_clean.csv")

print(f"Created {out_zip} with size {out_zip.stat().st_size / 1e6:.2f} MB")
