from deploy_to_sandbox import execute_remote

code = """
import gzip
import os

in_path = '/marimo/checkpoints/best_model_weights.pt'
out_path = '/marimo/checkpoints/best_model_weights.pt.gz'

with open(in_path, 'rb') as f_in, gzip.open(out_path, 'wb', compresslevel=9) as f_out:
    f_out.write(f_in.read())

print("Gzip size:", os.path.getsize(out_path) / 1024 / 1024, "MB")
"""
execute_remote(code)
