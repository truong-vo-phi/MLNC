import marimo._code_mode as cm

cell_code = """import torch
from src.models.ced_moe_split import CEDSingleMoESplitFFN, AcousticFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
moe_model = CEDSingleMoESplitFFN(
    num_classes=36,
    load_pretrained=True,
    pretrained_path="/marimo/pretrained/model.safetensors",
).to(device)

dummy_wav = torch.randn(1, 64000, device=device)
with torch.no_grad():
    res = moe_model.forward_full(dummy_wav)

p_info = moe_model.get_num_params()
f_spec = res["spectral_feat"][0].cpu().tolist()
alpha_val = res["alpha"].item()

mo.md(
    f\"\"\"
### \\U0001f680 CED-SplitFFN-MoE (6 Acoustic Features) Deployed Successfully!

- **Hardware:** `{device}` (NVIDIA RTX PRO 6000 Blackwell Server Edition)
- **Total Parameters:** `{p_info['total']:,}` (Backbone + SplitFFN MoE: `{p_info['trainable']:,}`, Router: `{p_info['router']:,}`)
- **AudioSet CED-Tiny Pretrained Weights:** `{moe_model.pretrained_loaded_tensors}` tensors loaded

#### \\U0001f4ca 6 Physical & Spectral Morphology Features:
1. **f_1 (Log-RMS Energy):** `{f_spec[0]:.4f}`
2. **f_2 (Crest Factor - Peak / RMS):** `{f_spec[1]:.4f}`
3. **f_3 (Temporal Kurtosis of Frame Energy):** `{f_spec[2]:.4f}`
4. **f_4 (Spectral Flatness / Wiener Entropy):** `{f_spec[3]:.4f}`
5. **f_5 (Low-band to Speech-band Energy Ratio <300Hz vs 300-3400Hz):** `{f_spec[4]:.4f}`
6. **f_6 (High-band to Speech-band Energy Ratio >3400Hz vs 300-3400Hz):** `{f_spec[5]:.4f}`

#### \\U0001f500 Router Weight & Classification Output:
- **Routing Weight High-SNR (\\\\alpha):** `{alpha_val:.4f}` ({alpha_val*100:.1f}%)
- **Routing Weight Low-SNR (1 - \\\\alpha):** `{1.0 - alpha_val:.4f}` ({(1.0 - alpha_val)*100:.1f}%)
- **Output Logits:** shape `{tuple(res['logits'].shape)}` for 36 noise classes.
\"\"\"
)
"""

async def apply_cell():
    async with cm.get_context() as ctx:
        _ = ctx.cells["vblA"].code
        ctx.edit_cell("vblA", code=cell_code, hide_code=False)
        ctx.run_cell("vblA")

await apply_cell()
print("vblA edited and executed!")
