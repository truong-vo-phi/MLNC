import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

code_cell11 = '''
# Khởi tạo mô hình hoàn chỉnh CEDSingleMoESplitFFN và kiểm chứng kiến trúc & số lượng tham số
model_champion = CEDSingleMoESplitFFN(
    num_classes=36, freq_bins=64, frames=398, split_blocks=(10, 11), load_pretrained=True
).to(device)
model_champion.eval()

# Phân rã chi tiết số lượng tham số từng thành phần
param_info = model_champion.get_num_params()
total_p = param_info["total"]
router_p = param_info["router"]
experts_p = 2 * 2 * (192 * 768 + 768 + 768 * 192 + 192)  # 2 khối MoE (10, 11) x 2 chuyên gia (Low, High)
heads_p = sum(p.numel() for n, p in model_champion.named_parameters() if "head_" in n)
backbone_shared_p = total_p - (experts_p + router_p + heads_p)

# Chạy thử nghiệm End-to-End với 2 đoạn âm thanh giả lập để kiểm chứng luồng tensor
_dummy_wave = torch.randn(2, 64000, device=device)
with torch.no_grad():
    _output = model_champion(_dummy_wave, return_dict=True)

print("=== THỐNG KÊ CHI TIẾT THAM SỐ KIẾN TRÚC CED-SingleMoE ===")
print(f"Tổng số tham số: {total_p:,} (~{total_p / 1e6:.2f}M | Dung lượng Checkpoint: ~{total_p * 4 / (1024**2):.2f} MB)")
print(f"  • Backbone Dùng Chung (Blocks 0-9 & Attn 10-11): {backbone_shared_p:,} ({backbone_shared_p/total_p*100:.2f}%)")
print(f"  • Chuyên Gia Phân Nhánh MoE (Blocks 10-11 Low/High): {experts_p:,} ({experts_p/total_p*100:.2f}%)")
print(f"  • Bộ Định Tuyến Tự Thân (Spectral-Token Router): {router_p:,} ({router_p/total_p*100:.2f}%)")
print(f"  • Đầu Phân Loại Kép (Dual Heads Low/High): {heads_p:,} ({heads_p/total_p*100:.2f}%)")
print(f"\\nKiểm tra luồng Tensor:")
print(f"  Waveform: {_dummy_wave.shape} -> Logits: {_output['logits'].shape} -> Alpha: {_output['alpha'].squeeze().tolist()}")

# Trình bày giao diện trực quan trong Marimo Notebook
mo.md(f"""
### 📊 Bảng Phân Bổ Tham Số & Kiến Trúc Mô Hình CED-SingleMoE-SplitFFN

| Thành Phần Kiến Trúc | Số Lớp / Cấu Trúc | Số Tham Số | Tỷ Lệ (%) | Vai Trò Kỹ Thuật |
| :--- | :--- | :---: | :---: | :--- |
| **Backbone Dùng Chung** | Blocks 0–9 + Attention 10–11 | **{backbone_shared_p:,}** | **{backbone_shared_p/total_p*100:.2f}%** | Trích xuất biểu diễn âm học đa tỉ lệ, 2D GREP BiasAttention |
| **Chuyên Gia Split-MoE** | 2 Khối $\\\\times$ 2 Experts (Low/High) | **{experts_p:,}** | **{experts_p/total_p*100:.2f}%** | Thích ứng hóa phi tuyến theo từng chế độ nhiễu |
| **Đầu Phân Loại Kép** | Head Low + Head High (192 $\\\\to$ 36) | **{heads_p:,}** | **{heads_p/total_p*100:.2f}%** | Tối ưu biên phân định cho từng phân vùng SNR |
| **Bộ Định Tuyến Router** | Token $h_9$ (16d) $\\\\parallel f_{{\\\\text{{spec}}}}$ (6d) $\\\\to$ MLP | **{router_p:,}** | **{router_p/total_p*100:.2f}%** | Ước lượng liên tục $\\\\alpha \\\\in (0, 1)$ mà không cần nhãn Oracle SNR |
| **TỔNG CỘNG** | **Toàn Bộ Mạng (12 Blocks ViT-MoE)** | **{total_p:,}** | **100.00%** | **~{total_p / 1e6:.2f}M Tham Số (Dung lượng: ~{total_p * 4 / (1024**2):.2f} MB)** |

> 💡 **Khởi tạo Warm-Start (Identity Step 0):** Tại thời điểm ban đầu, cả 3 nhánh FFN (Shared, Low, High) đều được sao chép nguyên vẹn từ trọng số tiền huấn luyện CED-Tiny. Vì hệ số tổng hợp là $0.5 + 0.5(1-\\\\alpha) + 0.5\\\\alpha \\\\equiv 1.0$, nên đầu ra khởi tạo đúng bằng CED-Tiny gốc với mọi $\\\\alpha$, hoàn toàn không gây sốc phân phối kích hoạt!
""")
'''

update_script = f"""
import marimo._code_mode as cm

async def update():
    async with cm.get_context() as ctx:
        ctx.edit_cell("Kclp", {repr(code_cell11.strip())})
        ctx.run_cell("Kclp")

await update()
"""

rc = execute_remote(update_script)
print(f"Cell 11 update executed with code {rc}")
