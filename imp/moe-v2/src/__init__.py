"""
Single-Backbone Mixture-of-Experts (MoE) v2 for Noise-Robust Audio Classification.

Theoretical Lineage:
- CED-Tiny: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957
- V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts", NeurIPS 2021. arXiv:2106.05974
- Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951
- DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE", ACL 2024. arXiv:2401.06066
- HydraLoRA: Tian et al., "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning", NeurIPS 2024. arXiv:2404.19245
- BEATs: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023. arXiv:2212.09058
"""

__version__ = "2.0.0"
