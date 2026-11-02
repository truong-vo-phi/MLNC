from deploy_to_sandbox import execute_remote

code = """
import os
import torch

ckpt = torch.load('/marimo/checkpoints/best_model.pt', map_location='cpu')
sd = ckpt['model_state_dict']

# Filter only trained parameters (router, lora, heads)
adapter_sd = {}
for k, v in sd.items():
    if 'lora_' in k or 'router' in k or 'head_low' in k or 'head_high' in k:
        adapter_sd[k] = v.cpu()

print(f"Total adapter keys: {len(adapter_sd)}")
total_params = sum(p.numel() for p in adapter_sd.values())
print(f"Total adapter parameters: {total_params:,} ({total_params / 1e6:.3f}M)")

adapter_path = '/marimo/checkpoints/best_adapter_weights.pt'
torch.save({
    'adapter_state_dict': adapter_sd,
    'best_top1_acc': ckpt.get('best_top1_acc', 0.0),
    'best_results': ckpt.get('best_results', {}),
}, adapter_path)

print('best_adapter_weights.pt saved! Size:', os.path.getsize(adapter_path) / 1024, 'KB')
"""
execute_remote(code)
