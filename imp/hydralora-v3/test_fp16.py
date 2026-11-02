from deploy_to_sandbox import execute_remote

code = """
import gzip
import os
import torch

ckpt = torch.load('/marimo/checkpoints/best_model_weights.pt', map_location='cpu')
sd_fp16 = {k: v.half() if v.is_floating_point() else v for k, v in ckpt['model_state_dict'].items()}

fp16_path = '/marimo/checkpoints/best_model_fp16.pt'
torch.save({
    'model_state_dict': sd_fp16,
    'best_top1_acc': ckpt.get('best_top1_acc', 0.0),
    'best_results': ckpt.get('best_results', {}),
}, fp16_path)

print("FP16 size:", os.path.getsize(fp16_path) / 1024 / 1024, "MB")
"""
execute_remote(code)
