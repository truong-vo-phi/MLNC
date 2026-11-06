import marimo._code_mode as cm

code = '''def print_dual_ced_tiny_architecture() -> None:
    arch_diagram = """
====================================================================================================
                        DUAL-CED-TINY ARCHITECTURE SPECIFICATION
                    (Dual-Expert Audio-Transformer with SNR Router)
====================================================================================================

1. TONG QUAN HE THONG (SYSTEM OVERVIEW):
   - Model Family: Dual-Backbone Mixture-of-Experts (MoE) Audio Classifier
   - Target: 36-class Environmental Audio Tagging under Multiple SNR levels (-5 dB to 20 dB)
   - Total Parameters: ~10.79M (2 independent CED-Tiny backbones x ~5.40M)
   - Routing Mechanism: EnergySNRRouter with SNR cutoff threshold tau = 18.0 dB

2. SO DO LUONG DU LIEU (END-TO-END DATAFLOW):

   Waveform x_mix: [B, 64000] (Mono, 16 kHz, 4.0 seconds)
              |
              +--> [EnergySNRRouter (tau = 18.0 dB)]:
              |      Estimate Signal-to-Noise Ratio (SNR in dB)
              |      -> alpha = 0.0 (if SNR < 18 dB) | alpha = 1.0 (if SNR >= 18 dB)
              |
              +--> [LogMel Frontend (CED-spec)]:
                     STFT (n_fft=512, win=400, hop=160) -> 64 Mel bins, 20Hz - 8000Hz
                     Log-compression & Z-score Instance Normalization
                     -> Spectrogram Tensor: [B, 1, 64, 398]
                            |
              +-------------+-------------+
              |                           |
              v                           v
   [S1: CED-Tiny Low-Expert]   [S2: CED-Tiny High-Expert]
   (5,395,722 params ~ 5.40M)  (5,395,722 params ~ 5.40M)
   * PatchEmbed 16x16          * PatchEmbed 16x16
   * 12 CED Blocks (D=192,H=3) * 12 CED Blocks (D=192,H=3)
   * 2D Relative Pos Bias      * 2D Relative Pos Bias
   * Specialized for Low-SNR   * Specialized for High-SNR
   * Linear Head (36 classes)  * Linear Head (36 classes)
              |                           |
              v                           v
      Logits S1 [B, 36]           Logits S2 [B, 36]
              |                           |
              +-------------+-------------+
                            |
                            v
              [Dynamic Gating / Blending]:
              z_final = (1 - alpha) * z_low + alpha * z_high
                            |
                            v
               Final Output Logits: [B, 36]

3. BANG PHAN BO THAM SO (PARAMETER BUDGET):
   +------------------------------------+----------------+----------------+
   | Component / Module                 | Parameter Count| Percentage (%) |
   +------------------------------------+----------------+----------------+
   | S1: CED-Tiny Low-Expert Backbone   |   5,395,722    |     50.0%      |
   |   - Patch Embedding (16x16)        |      49,344    |      0.5%      |
   |   - 12x Transformer Blocks (D=192) |   5,338,368    |     49.5%      |
   |   - Classifier Head (192 -> 36)    |       6,948    |      0.1%      |
   | S2: CED-Tiny High-Expert Backbone  |   5,395,722    |     50.0%      |
   |   - Patch Embedding (16x16)        |      49,344    |      0.5%      |
   |   - 12x Transformer Blocks (D=192) |   5,338,368    |     49.5%      |
   |   - Classifier Head (192 -> 36)    |       6,948    |      0.1%      |
   | Router: EnergySNRRouter (tau=18dB) |           0    |      0.0% (Non-param)
   +------------------------------------+----------------+----------------+
   | TOTAL DUAL-CED-TINY SYSTEM         |  10,791,444    |    100.0% (~10.8M)
   +------------------------------------+----------------+----------------+
====================================================================================================
"""
    print(arch_diagram)

print_dual_ced_tiny_architecture()
'''

async with cm.get_context() as ctx:
    ctx.edit_cell("nGLA", code=code, hide_code=False, name="architecture_display")
    ctx.run_cell("nGLA")
print("Updated nGLA successfully")
