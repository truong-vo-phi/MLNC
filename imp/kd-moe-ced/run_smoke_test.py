"""
Comprehensive Smoke-Test Script for CED-SingleMoE-SplitFFN
Kiểm tra toàn diện tất cả các thành phần:
1. Frontend & 6 Acoustic Features
2. ViTSBias Backbone & 2D GREP BiasAttention
3. Self-Contained Spectral-Token Router
4. SplitMoEFFN (Blocks 10-11) & Identity Warm-Start
5. Dual Classifier Heads & Soft Blending
6. Full Forward & Backward Gradient Flow
7. ChampionPipeline with 3-Way TTA
8. 9 Metrics Computation
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.frontend import (
    TARGET_RMS,
    AcousticFeatureExtractor,
    LogMel,
    SpecAugment,
    rms_normalize,
)
from src.router import SelfContainedSpectralTokenRouter
from src.models.vit_grep import ViTSBias, BiasAttention, SharedBias
from src.models.split_moe import CEDSingleMoESplitFFN, SplitMoEFFN
from src.metrics import compute_9_metrics_dict, compute_snr_breakdown
from src.pipeline import ChampionPipeline

def run_smoke_test():
    print("=" * 80)
    print("BẮT ĐẦU SMOKE-TEST KIỂM CHỨNG TOÀN DIỆN CED-SINGLEMOE-SPLITFFN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Thiết bị kiểm thử: {device}")
    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------------------------------------------------------
    # 1. FRONTEND VÀ 6 ĐẶC TRƯNG ÂM HỌC
    # --------------------------------------------------------------------------
    print("\n[1/7] Kiểm tra Frontend & 6 Đặc Trưng Âm Học...")
    dummy_wave = torch.randn(2, 64000, device=device)
    norm_wave = rms_normalize(dummy_wave, target_rms=TARGET_RMS)
    rms_val = norm_wave[0].square().mean().sqrt().item()
    assert abs(rms_val - TARGET_RMS) < 1e-4, f"RMS normalization lệch: {rms_val}"
    print(f"  ✓ RMS Normalization chuẩn xác: {rms_val:.5f} (Kỳ vọng: {TARGET_RMS})")

    extractor = AcousticFeatureExtractor().to(device)
    feats = extractor(norm_wave)
    assert feats.shape == (2, 6), f"Kích thước đặc trưng sai: {feats.shape}"
    print(f"  ✓ 6 Đặc trưng âm học vật lý: {feats.shape} [Batch=2, Dim=6]")

    frontend = LogMel().to(device)
    specaug = SpecAugment().to(device)
    mels = frontend(norm_wave)
    assert mels.shape == (2, 1, 64, 398), f"Kích thước Log-Mel sai: {mels.shape}"
    mels_aug = specaug(mels)
    assert mels_aug.shape == (2, 1, 64, 398)
    print(f"  ✓ Log-Mel HTK & SpecAugment: {mels.shape} [B=2, C=1, F=64, T=398]")

    # --------------------------------------------------------------------------
    # 2. BACKBONE VITSBIAS & 2D GREP BIASATTENTION
    # --------------------------------------------------------------------------
    print("\n[2/7] Kiểm tra Backbone ViTSBias & 2D GREP BiasAttention...")
    vit = ViTSBias(num_classes=36, pretrained=False).to(device)
    tokens = vit.vit.patch_embed(mels[..., :vit.frames])
    assert tokens.shape == (2, 96, 192), f"Patch token sai: {tokens.shape}"
    print(f"  ✓ Conv2d 16x16 Patch Embed: {tokens.shape} (96 patch tokens, 192 dim)")

    cls = vit.vit.cls_token.expand(2, -1, -1)
    seq = torch.cat([cls, tokens], dim=1) + vit.vit.pos_embed
    assert seq.shape == (2, 97, 192), f"Tokens + CLS sai: {seq.shape}"
    print(f"  ✓ Chuỗi token X_0 (kèm CLS): {seq.shape} (97 tokens, 192 dim)")

    # --------------------------------------------------------------------------
    # 3. ROUTER TỰ THÂN (SELF-CONTAINED SPECTRAL TOKEN ROUTER)
    # --------------------------------------------------------------------------
    print("\n[3/7] Kiểm tra Router Tự Thân (Spectral-Token Router)...")
    router = SelfContainedSpectralTokenRouter().to(device)
    cls9 = torch.randn(2, 192, device=device)
    alpha, gates = router(feats, cls9)
    assert alpha.shape == (2, 1) and gates.shape == (2, 2)
    assert torch.all((alpha >= 0.0) & (alpha <= 1.0))
    assert torch.allclose(gates.sum(dim=-1), torch.ones(2, device=device), atol=1e-5)
    print(f"  ✓ Router forward hợp lệ: alpha={alpha.squeeze().tolist()}, sum(gates)=1.0")

    # --------------------------------------------------------------------------
    # 4. SPLIT-MOE FFN (BLOCKS 10-11) & IDENTITY WARM-START
    # --------------------------------------------------------------------------
    print("\n[4/7] Kiểm tra SplitMoEFFN & Tính Bảo Toàn Trọng Số (Warm-Start)...")
    split_ffn = SplitMoEFFN(base_mlp=vit.vit.blocks[0].mlp).to(device)
    u = torch.randn(2, 97, 192, device=device)
    out_warm = split_ffn(u, alpha)
    out_shared = split_ffn.ffn_shared(u)
    diff = (out_warm - out_shared).abs().max().item()
    assert diff < 1e-5, f"Warm-start không bảo toàn identity: diff={diff}"
    print(f"  ✓ Identity Warm-Start Step 0 bảo toàn 100% (max diff = {diff:.2e})")

    # --------------------------------------------------------------------------
    # 5. MÔ HÌNH TOÀN DIỆN CEDSINGLEMOESPLITFFN & PARAMETER COUNT
    # --------------------------------------------------------------------------
    print("\n[5/7] Kiểm tra Mô Hình Hoàn Chỉnh CEDSingleMoESplitFFN...")
    model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=False).to(device)
    p_info = model.get_num_params()
    total_params = p_info["total"]
    router_params = p_info["router"]
    assert total_params == 6622599, f"Số tham số sai: {total_params}"
    assert router_params == 4338, f"Số tham số router sai: {router_params}"
    print(f"  ✓ Tổng số tham số mạng: {total_params:,} (~6.62M | Router: {router_params:,} ~0.07%)")

    # --------------------------------------------------------------------------
    # 6. LAN TRUYỀN XUÔI & NGƯỢC (FORWARD & BACKWARD GRADIENT FLOW)
    # --------------------------------------------------------------------------
    print("\n[6/7] Kiểm tra Lan Truyền Xuôi & Ngược (Forward & Backward Gradient)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    out_dict = model(dummy_wave, return_dict=True)
    logits = out_dict["logits"]
    logits_low = out_dict["logits_low"]
    logits_high = out_dict["logits_high"]
    alpha_pred = out_dict["alpha"]

    target_labels = torch.zeros(2, 36, device=device)
    target_labels[:, 0] = 1.0
    dummy_snr = torch.tensor([-5.0, 15.0], device=device)

    loss_hard = nn.BCEWithLogitsLoss()(logits, target_labels)
    loss_router = model.router.compute_router_loss(alpha_pred, dummy_snr)
    total_loss = loss_hard + 0.1 * loss_router
    total_loss.backward()

    # Kiểm tra gradient của tất cả các nhánh
    assert model.vit_base.vit.patch_embed.proj.weight.grad is not None
    assert model.router.mlp[0].weight.grad is not None
    assert model.head_low.weight.grad is not None
    assert model.head_high.weight.grad is not None
    optimizer.step()
    print(f"  ✓ Lan truyền ngược Gradient 100% thông suốt: Loss = {total_loss.item():.4f}")

    # --------------------------------------------------------------------------
    # 7. CHAMPION PIPELINE V2 VÀ 9 METRICS BREAKDOWN
    # --------------------------------------------------------------------------
    print("\n[7/7] Kiểm tra Champion Pipeline v2 & 9 Metrics Breakdown...")
    model.eval()
    pipeline = ChampionPipeline(student_model=model, device=device)
    probs_tta, alphas_tta = pipeline.predict_student_tta(dummy_wave)
    assert probs_tta.shape == (2, 36) and alphas_tta.shape == (2, 1)
    print(f"  ✓ 3-Way Temporal TTA (±50ms) suy luận hợp lệ: {probs_tta.shape}")

    # Kiểm tra tính toán 9 metrics
    dummy_preds = np.random.dirichlet(np.ones(36), size=12)
    dummy_targets = np.random.randint(0, 36, size=12)
    dummy_snrs = np.array([-5, -5, 0, 0, 5, 5, 10, 10, 15, 15, 20, 20])
    dummy_alphas = np.random.uniform(0.4, 0.7, size=12)

    snr_breakdown = compute_snr_breakdown(dummy_preds, dummy_targets, dummy_snrs, dummy_alphas)
    assert "all" in snr_breakdown and "-5dB" in snr_breakdown
    assert len(snr_breakdown["all"]) >= 9
    print(f"  ✓ Đầy đủ 9 Metrics theo từng mốc SNR và ALL:")
    for k in ["top1", "top3", "balanced_acc", "map", "auc", "precision", "recall", "f1", "mean_alpha"]:
        assert k in snr_breakdown["all"], f"Thiếu metric {k}"
    print("    [Top-1, Top-3, Balanced-Acc, mAP, AUC, Precision, Recall, F1, Alpha]")

    print("\n" + "=" * 80)
    print("🎉 TẤT CẢ CÁC BƯỚC SMOKE-TEST ĐÃ THÀNH CÔNG RỰC RỠ 100%!")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
