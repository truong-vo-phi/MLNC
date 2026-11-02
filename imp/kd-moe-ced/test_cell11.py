import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

test_code = """
import torch
import pandas as pd
from src.models.split_moe import CEDSingleMoESplitFFN

model = CEDSingleMoESplitFFN(num_classes=36, freq_bins=64, frames=398, split_blocks=(10, 11), load_pretrained=True).eval()
param_info = model.get_num_params()

# Detailed Parameter Breakdown
backbone_shared_params = sum(p.numel() for name, p in model.vit_base.named_parameters() if 'blocks.10' not in name and 'blocks.11' not in name and 'head' not in name)
moe_shared_params = sum(p.numel() for name, p in model.named_parameters() if 'ffn_shared' in name)
moe_routed_params = sum(p.numel() for name, p in model.named_parameters() if 'ffn_low' in name or 'ffn_high' in name)
router_params = sum(p.numel() for p in model.router.parameters())
dual_head_params = sum(p.numel() for p in model.head_low.parameters()) + sum(p.numel() for p in model.head_high.parameters())
total_params = sum(p.numel() for p in model.parameters())

print(f"Backbone Base (Blocks 0..9 + Attn 10..11): {backbone_shared_params:,} ({backbone_shared_params/total_params:.2%})")
print(f"MoE Shared FFN (Blocks 10..11): {moe_shared_params:,} ({moe_shared_params/total_params:.2%})")
print(f"MoE Routed FFNs (Low + High): {moe_routed_params:,} ({moe_routed_params/total_params:.2%})")
print(f"Self-Contained Router: {router_params:,} ({router_params/total_params:.2%})")
print(f"Dual Classifier Heads: {dual_head_params:,} ({dual_head_params/total_params:.2%})")
print(f"TOTAL PARAMETERS: {total_params:,} (~{total_params/1e6:.2f}M)")
"""

execute_remote(test_code)
