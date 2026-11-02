from deploy_to_sandbox import execute_remote

code = """
import os
import torch

ckpt = torch.load('/marimo/checkpoints/best_model.pt', map_location='cpu')
weights_dict = {
    'model_state_dict': ckpt['model_state_dict'],
    'best_top1_acc': ckpt.get('best_top1_acc', 0.0),
    'best_results': ckpt.get('best_results', {}),
}
target_path = '/marimo/checkpoints/best_model_weights.pt'
torch.save(weights_dict, target_path)
print('best_model_weights.pt saved! Size:', os.path.getsize(target_path) / 1024 / 1024, 'MB')
"""
execute_remote(code)
