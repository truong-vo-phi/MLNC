# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "timm==1.0.30",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # KD teacher-student: BEATs (nhiễu sạch) → FRCRN + MobileViT-S, học bằng q/k/v và CRD

    | Nhánh | Đường đi | Trạng thái |
    | --- | --- | --- |
    | Teacher | **nhiễu thật** → chuẩn hoá RMS → fbank → BEATs fine-tune (`pretrained/beat-ft`, fold 0) | đóng băng |
    | Student | mixture → FRCRN (`pretrained/FRCRN/best_e037_…pt`, mask) → n̂ → log-mel → **MobileViT 12 lớp** (L = 2, 6, 4; d = 768; 12 head) → attention pooling → logit 36 | FRCRN đóng băng, MobileViT huấn luyện |
    | KD | **q/k/v** của 12 lớp self-attention BEATs ← 12 lớp Transformer của student (khối 1: lớp 1-2 ↔ BEATs 1-2, khối 2: lớp 1-6 ↔ BEATs 3-8, khối 3: lớp 1-4 ↔ BEATs 9-12); **CRD** trên embedding đã pool; **logit** 36 lớp của head teacher | chỉ lúc train |

    **Loss** = CE (label smoothing 0.1) + λ_qkv · L_qkv + λ_crd · L_crd + λ_logit · L_logit, các λ tăng tuyến tính trong `warmup_epochs` epoch đầu.

    - **L_qkv** (mỗi cặp lớp, mỗi loại q, k, v): `1 − cos` theo từng token + `REL_WEIGHT` · KL giữa ma trận quan hệ `softmax(xxᵀ/√d)` theo từng head (kiểu MiniLM). Cosine không phụ thuộc thang đo, nên q, k, v có độ lớn khác nhau vẫn so sánh được. Student có cùng d = 768 và 12 head với BEATs nên so q, k, v trực tiếp, không cần adapter. Student lấy q/k/v từ `attn.qkv` của từng lớp (trung bình theo 4 vị trí trong patch); q/k/v của teacher được pool trung bình về lưới token của khối student (8×25, 4×13, 2×7 với clip 4 s).
    - **L_crd** (Tian et al., 2020): hai head chiếu về 128 chiều, L2-norm. Positive là cùng mẫu ở hai view, negative lấy `CRD_NEGATIVES` mẫu ngẫu nhiên từ memory bank (cập nhật momentum). Negative cùng lớp với anchor bị loại, để tránh false negative vì tập chỉ có 36 lớp.
    - **L_logit** (Hinton et al., 2015): `T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))` với T = 4. Teacher nhìn nhiễu sạch nên logit của nó là nhãn mềm "có đặc quyền"; đổi lại teacher fold 0 chỉ đạt khoảng 0.74 accuracy, nên nhãn mềm có nhiễu và CE vẫn giữ vai trò neo.
    - **Đánh giá:** 9 chỉ số (top1, top3, balanced acc, precision, recall, macro-F1, micro-F1, mAP, macro-AUC) tại từng mức SNR (−5…20) và trung bình. Checkpoint chọn theo `macro_acc` (balanced acc trung bình theo SNR), đúng chuẩn repo.

    `RUN_TRAINING = False` chỉ chạy smoke test nhỏ (64 mẫu, 1 epoch) vào `output/test/`. Cell (1) có cờ `USE_QKV`, `USE_CRD`, `USE_LOGIT` để ablation: CE → +logit → +qkv → +crd → +cả ba.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center"><img alt="Tổng quan pipeline" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5MDAiIGhlaWdodD0iNDIwIiB2aWV3Qm94PSIwIDAgOTAwIDQyMCI+PHRpdGxlPlThu5VuZyBxdWFuIHBpcGVsaW5lIEtEPC90aXRsZT48ZGVmcz48bWFya2VyIGlkPSJhcnIiIHZpZXdCb3g9IjAgMCAxMCAxMCIgcmVmWD0iOSIgcmVmWT0iNSIgbWFya2VyV2lkdGg9IjciIG1hcmtlckhlaWdodD0iNyIgb3JpZW50PSJhdXRvLXN0YXJ0LXJldmVyc2UiPjxwYXRoIGQ9Ik0wIDAgTDEwIDUgTDAgMTAgeiIgZmlsbD0iIzRiNTU2MyIvPjwvbWFya2VyPjwvZGVmcz48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjQyMCIgZmlsbD0iI2ZmZmZmZiIvPjxyZWN0IHg9IjEwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iMzIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+VGVhY2hlcjwvdGV4dD48cmVjdCB4PSIxNjAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjQiLz48dGV4dCB4PSIxODIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+U3R1ZGVudDwvdGV4dD48cmVjdCB4PSIzMTAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjQiLz48dGV4dCB4PSIzMzIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+TG9zcyBLRDwvdGV4dD48cmVjdCB4PSI0NjAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNmM2Y0ZjYiIHN0cm9rZT0iIzZiNzI4MCIgc3Ryb2tlLXdpZHRoPSIxLjQiIHN0cm9rZS1kYXNoYXJyYXk9IjQgMyIvPjx0ZXh0IHg9IjQ4MiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj7EkMOzbmcgYsSDbmcgKG7DqXQgxJHhu6l0KTwvdGV4dD48cmVjdCB4PSIxMCIgeT0iNDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM2YjcyODAiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNjAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+Tmhp4buFdSB0aOG6rXQ8L3RleHQ+PHRleHQgeD0iNjAuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+KEIsIDY0MDAwKTwvdGV4dD48cmVjdCB4PSIxNDAiIHk9IjQwIiB3aWR0aD0iMTMwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjIwNS4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5STVMgKyBmYmFuazwvdGV4dD48dGV4dCB4PSIyMDUuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+Uk1TIDAuMDc5MjU8L3RleHQ+PHJlY3QgeD0iMzAwIiB5PSI0MCIgd2lkdGg9IjE5MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkYmVhZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1kYXNoYXJyYXk9IjYgNCIvPjx0ZXh0IHg9IjM5NS4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5CRUFUcyBmb2xkIDA8L3RleHQ+PHRleHQgeD0iMzk1LjAiIHk9IjgwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjEyIGzhu5twLCBmaW5lLXR1bmUgMzYgbOG7m3A8L3RleHQ+PHBhdGggZD0iTTExMCA2OCBMMTQwIDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNMjcwIDY4IEwzMDAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjMwMCIgeT0iMTY1IiB3aWR0aD0iMTkwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjM5NS4wIiB5PSIxODkuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TF9xa3Y8L3RleHQ+PHRleHQgeD0iMzk1LjAiIHk9IjIwNS4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5jb3NpbmUgKyBLTCBxdWFuIGjhu4c8L3RleHQ+PHJlY3QgeD0iNTMwIiB5PSIxNjUiIHdpZHRoPSIxNzAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNjE1LjAiIHk9IjE4OS43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MX2NyZDwvdGV4dD48dGV4dCB4PSI2MTUuMCIgeT0iMjA1LjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPkluZm9OQ0UgKyBtZW1vcnkgYmFuazwvdGV4dD48cmVjdCB4PSI3MzAiIHk9IjE2NSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI4MDUuMCIgeT0iMTg5Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkxfbG9naXQgKHThuq90KTwvdGV4dD48dGV4dCB4PSI4MDUuMCIgeT0iMjA1LjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPktMLCBUID0gNDwvdGV4dD48cGF0aCBkPSJNMzk1IDk2IEwzOTUgMTY1IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSI0NzAiIHk9IjEzNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZmlsbD0iIzM3NDE1MSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+cSwgaywgdiBj4bunYSAxMiBs4bubcDwvdGV4dD48cGF0aCBkPSJNNDkwIDY4IEw2MTUgNjggTDYxNSAxNjUiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjx0ZXh0IHg9IjU1MiIgeT0iNjAiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmVtYmVkZGluZyA3Njg8L3RleHQ+PHBhdGggZD0iTTQ5MCA4MiBMODA1IDgyIEw4MDUgMTY1IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSI3MjAiIHk9Ijc2IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmaWxsPSIjMzc0MTUxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5sb2dpdCAzNiAoaGVhZCB0ZWFjaGVyKTwvdGV4dD48cmVjdCB4PSIxMCIgeT0iMzAwIiB3aWR0aD0iOTAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM2YjcyODAiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNTUuMCIgeT0iMzI0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPk1peHR1cmU8L3RleHQ+PHRleHQgeD0iNTUuMCIgeT0iMzQwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCLCA2NDAwMCk8L3RleHQ+PHJlY3QgeD0iMTI1IiB5PSIzMDAiIHdpZHRoPSIxMTUiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42IiBzdHJva2UtZGFzaGFycmF5PSI2IDQiLz48dGV4dCB4PSIxODIuNSIgeT0iMzI0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkZSQ1JOIChtYXNrKTwvdGV4dD48dGV4dCB4PSIxODIuNSIgeT0iMzQwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmJlc3RfZTAzNzwvdGV4dD48cmVjdCB4PSIyNjUiIHk9IjMwMCIgd2lkdGg9IjgwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjMwNS4wIiB5PSIzMjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+bG9nLW1lbDwvdGV4dD48dGV4dCB4PSIzMDUuMCIgeT0iMzQwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjEyOCDDlyAzOTg8L3RleHQ+PHJlY3QgeD0iMzc1IiB5PSIzMDAiIHdpZHRoPSIyMjAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNDg1LjAiIHk9IjMyNC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5Nb2JpbGVWaVQgMTIgbOG7m3A8L3RleHQ+PHRleHQgeD0iNDg1LjAiIHk9IjM0MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5MID0gMiwgNiwgNCDCtyBkID0gNzY4IMK3IDEyIGhlYWQ8L3RleHQ+PHJlY3QgeD0iNjI1IiB5PSIzMDAiIHdpZHRoPSI5MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI2NzAuMCIgeT0iMzI0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkxvZ2l0PC90ZXh0Pjx0ZXh0IHg9IjY3MC4wIiB5PSIzNDAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+YXR0LiBwb29saW5nPC90ZXh0PjxyZWN0IHg9Ijc0NSIgeT0iMzAwIiB3aWR0aD0iOTAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNzkwLjAiIHk9IjMyNC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5DRTwvdGV4dD48dGV4dCB4PSI3OTAuMCIgeT0iMzQwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmxhYmVsIHNtb290aCAwLjE8L3RleHQ+PHBhdGggZD0iTTEwMCAzMjggTDEyNSAzMjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik0yNDAgMzI4IEwyNjUgMzI4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSIyNTMiIHk9IjMyMSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZmlsbD0iIzM3NDE1MSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+bsyCPC90ZXh0PjxwYXRoIGQ9Ik0zNDUgMzI4IEwzNzUgMzI4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNTk1IDMyOCBMNjI1IDMyOCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTcxNSAzMjggTDc0NSAzMjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik00MTAgMzAwIEw0MTAgMjIxIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSIzNTIiIHk9IjI2MiIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZmlsbD0iIzM3NDE1MSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+cSwgaywgdiAxMiBs4bubcDwvdGV4dD48cGF0aCBkPSJNNTcwIDMwMCBMNTcwIDIyMSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHRleHQgeD0iNjQwIiB5PSIyNjIiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZpbGw9IiMzNzQxNTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmVtYmVkZGluZyA3Njg8L3RleHQ+PHBhdGggZD0iTTY4MCAzMDAgTDY4MCAyNzAgTDgwNSAyNzAgTDgwNSAyMjEiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjx0ZXh0IHg9Ijc0NSIgeT0iMjY0IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmaWxsPSIjMzc0MTUxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5sb2dpdDwvdGV4dD48dGV4dCB4PSI0NTAiIHk9IjM5OCIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMTExODI3Ij5Mb3NzID0gQ0UgKyDOu19xa3YgwrcgTF9xa3YgKyDOu19jcmQgwrcgTF9jcmQgKyDOu19sb2dpdCDCtyBMX2xvZ2l0ICAgKM67IHTEg25nIHR1eeG6v24gdMOtbmggdHJvbmcgNSBlcG9jaCDEkeG6p3UpPC90ZXh0Pjwvc3ZnPg=="/></p>

    <p align="center"><em>Hình 1. Tổng quan: teacher nhận nhiễu thật, student nhận mixture, hai loss KD nối hai nhánh.</em></p>
    """)
    return


@app.cell
def _():
    # (0) Import
    import copy
    import math
    import sys
    from collections import namedtuple
    from dataclasses import dataclass, replace
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import timm
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
    from torch.optim import AdamW
    from torch.utils.data import Dataset
    from tqdm.auto import tqdm

    ROOT = Path.cwd()
    if ROOT.name == "notebook":
        ROOT = ROOT.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (
        AdamW,
        Dataset,
        F,
        Path,
        ROOT,
        Tensor,
        copy,
        math,
        namedtuple,
        nn,
        pd,
        torch,
        tqdm,
    )


@app.cell
def _(ROOT):
    # (1) Cấu hình thí nghiệm. Đổi MỘT cờ mỗi lần để ablation.
    EXPERIMENT = "beat-mv-kd"
    RUN_TRAINING = True           # False: smoke test nhỏ vào output/test
    USE_QKV = True                # L_qkv
    USE_CRD = True                # L_crd
    USE_LOGIT = False             # L_logit (KL trên logit 36 lớp của teacher); lượt chạy này tắt, chỉ giữ q/k/v và CRD
    USE_DRIVE = True              # backup best.pt, resume.pt, history, log lên Google Drive (đăng nhập ở cell 2b)
    DRIVE_DIR = "FA26_Capstone/Truong/beat-mv-kd"
    NUM_WORKERS = 8

    BEAT_FOLD = 0
    BEAT_FT_PATH = ROOT / "pretrained" / "beat-ft" / f"beats_teacher_fold{BEAT_FOLD}_weights.pt"
    FRCRN_PATH = ROOT / "pretrained" / "FRCRN" / "best_e037_20261020T203958Z.pt"
    BEATS_CONFIG = ROOT / "config" / "beat.yaml"
    FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"

    TEACHER_LAYERS = tuple(range(1, 13))  # cả 12 lớp BEATs (đếm từ 1) lấy q/k/v
    # MobileViT 12 lớp: khối 1 (2 lớp) <-> BEATs 1-2, khối 2 (6 lớp) <-> BEATs 3-8, khối 3 (4 lớp) <-> BEATs 9-12
    STUDENT = dict(dim=768, heads=12, ffn_dim=1536, layers=(2, 6, 4), channels=(96, 128, 160), patch=2)
    assert sum(STUDENT["layers"]) == len(TEACHER_LAYERS) and STUDENT["dim"] == 768 and STUDENT["heads"] == 12
    TARGET_RMS = 0.07925          # chuẩn hoá RMS nhiễu trước fbank, như reviewkd.yaml
    NUM_CLASSES = 36
    SNR_LEVELS = (-5, 0, 5, 10, 15, 20)

    KD = dict(
        qkv_weight=1.0,           # λ_qkv (chưa tinh chỉnh)
        rel_weight=1.0,           # trọng số KL quan hệ so với cosine bên trong L_qkv
        crd_weight=0.5,           # λ_crd (chưa tinh chỉnh)
        logit_weight=1.0,         # λ_logit (chưa tinh chỉnh)
        logit_temperature=4.0,    # như loss.kd.temperature trong beat.yaml
        crd_dim=128,
        crd_negatives=4096,
        crd_temperature=0.07,
        crd_momentum=0.5,
        warmup_epochs=5,
    )

    FRONTEND = dict(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=128,
                    f_min=20.0, f_max=8000.0, num_frames=398, log_eps=1e-6, norm_eps=1e-5)
    SPECAUG = dict(freq_masks=2, freq_width=16, time_masks=2, time_width=24)

    TRAIN = dict(lr=3e-4, weight_decay=0.05, betas=(0.9, 0.999), warmup_ratio=0.05, min_lr_ratio=0.0,
                 label_smoothing=0.1, grad_clip=1.0, batch_size=32, eval_batch_size=64,
                 epochs=50, patience=15, dropout=0.1, seed=42)
    return (
        BEATS_CONFIG,
        BEAT_FT_PATH,
        DRIVE_DIR,
        EXPERIMENT,
        FRCRN_CONFIG,
        FRCRN_PATH,
        FRONTEND,
        KD,
        NUM_CLASSES,
        NUM_WORKERS,
        RUN_TRAINING,
        SNR_LEVELS,
        SPECAUG,
        STUDENT,
        TARGET_RMS,
        TEACHER_LAYERS,
        TRAIN,
        USE_CRD,
        USE_DRIVE,
        USE_LOGIT,
        USE_QKV,
    )


@app.cell
def _(
    EXPERIMENT,
    Path,
    ROOT,
    RUN_TRAINING,
    USE_CRD,
    USE_LOGIT,
    USE_QKV,
    torch,
):
    # (2) Đường dẫn đầu ra, logger, thiết bị
    from src.utils.logger import setup_logger
    from src.utils.paths import RunPaths
    from src.utils.seed import set_seed

    output_dir = Path("output") if RUN_TRAINING else Path("output") / "test"
    run = RunPaths.new(EXPERIMENT if RUN_TRAINING else f"{EXPERIMENT}-smoke", ROOT / output_dir)
    run.create()
    logger = setup_logger(EXPERIMENT, run.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    logger.info("run %s | device %s | RUN_TRAINING=%s | qkv=%s crd=%s logit=%s", run.run_id, device, RUN_TRAINING, USE_QKV, USE_CRD, USE_LOGIT)
    return amp, device, logger, run, set_seed


@app.cell
def _(DRIVE_DIR, RUN_TRAINING, USE_DRIVE, logger):
    # (2b) Google Drive: lần đầu hiện liên kết đăng nhập, đăng nhập xong cell mới chạy tiếp. fs = None khi tắt hoặc smoke test.
    fs = None
    if USE_DRIVE and RUN_TRAINING:
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    logger.info("drive: %s", f"bật, thư mục {DRIVE_DIR}" if fs is not None else "tắt")
    return (fs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Teacher: nhiễu → BEATs (fold 0, đóng băng)

    Checkpoint `beat-ft` lưu theo tên khoá BEATs gốc (`encoder.model.*`, `head.*`), nên cần đổi tên sang `BEATClassifier`. `convert_checkpoint` của repo lo phần `backbone.*`, `fc1/fc2` và `patch_embedding`; head đổi tay thành `head.fc.*`. Nạp xong kiểm tra không thiếu hay thừa khoá nào.

    q/k/v lấy bằng forward hook trên `q_proj/k_proj/v_proj` của `GatedRelPosAttention`, tức là trước khi chia head. Mỗi tensor có dạng (B, N, 768) với N = 8 × 24 token, thứ tự (thời gian, tần số), cùng thứ tự với `tokens_to_grid` của repo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center"><img alt="Teacher BEATs" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAwIiBoZWlnaHQ9IjM2MCIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj48dGl0bGU+VGVhY2hlciBCRUFUcyB2w6AgxJFp4buDbSBs4bqleSBxLCBrLCB2IGPhu6dhIGPhuqMgMTIgbOG7m3A8L3RpdGxlPjxkZWZzPjxtYXJrZXIgaWQ9ImFyciIgdmlld0JveD0iMCAwIDEwIDEwIiByZWZYPSI5IiByZWZZPSI1IiBtYXJrZXJXaWR0aD0iNyIgbWFya2VySGVpZ2h0PSI3IiBvcmllbnQ9ImF1dG8tc3RhcnQtcmV2ZXJzZSI+PHBhdGggZD0iTTAgMCBMMTAgNSBMMCAxMCB6IiBmaWxsPSIjNGI1NTYzIi8+PC9tYXJrZXI+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAwIiBoZWlnaHQ9IjM2MCIgZmlsbD0iI2ZmZmZmZiIvPjxyZWN0IHg9IjEwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iMzIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+VGVhY2hlcjwvdGV4dD48cmVjdCB4PSIxNjAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNmZWY5YzMiIHN0cm9rZT0iI2NhOGEwNCIgc3Ryb2tlLXdpZHRoPSIxLjQiLz48dGV4dCB4PSIxODIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+TOG6pXkgcSwgaywgdjwvdGV4dD48cmVjdCB4PSIzMTAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjQiLz48dGV4dCB4PSIzMzIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+RMO5bmcgY2hvIGxvc3M8L3RleHQ+PHJlY3QgeD0iNDYwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM2YjcyODAiIHN0cm9rZS13aWR0aD0iMS40IiBzdHJva2UtZGFzaGFycmF5PSI0IDMiLz48dGV4dCB4PSI0ODIiIHk9IjE4LjUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+xJDDs25nIGLEg25nPC90ZXh0PjxyZWN0IHg9IjEwIiB5PSI0MCIgd2lkdGg9IjkwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2YzZjRmNiIgc3Ryb2tlPSIjNmI3MjgwIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjU1LjAiIHk9IjY0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPk5oaeG7hXUgdGjhuq10PC90ZXh0Pjx0ZXh0IHg9IjU1LjAiIHk9IjgwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCLCA2NDAwMCk8L3RleHQ+PHJlY3QgeD0iMTI1IiB5PSI0MCIgd2lkdGg9IjExMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkYmVhZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1kYXNoYXJyYXk9IjYgNCIvPjx0ZXh0IHg9IjE4MC4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5DaHXhuqluIGhvw6EgUk1TPC90ZXh0Pjx0ZXh0IHg9IjE4MC4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj50YXJnZXQgMC4wNzkyNTwvdGV4dD48cmVjdCB4PSIyNjAiIHk9IjQwIiB3aWR0aD0iMTEwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNiIgc3Ryb2tlLWRhc2hhcnJheT0iNiA0Ii8+PHRleHQgeD0iMzE1LjAiIHk9IjY0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkZiYW5rPC90ZXh0Pjx0ZXh0IHg9IjMxNS4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj4oQiwgMzk4LCAxMjgpPC90ZXh0PjxyZWN0IHg9IjM5NSIgeT0iNDAiIHdpZHRoPSIxMzAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS42IiBzdHJva2UtZGFzaGFycmF5PSI2IDQiLz48dGV4dCB4PSI0NjAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+UGF0Y2ggZW1iZWQ8L3RleHQ+PHRleHQgeD0iNDYwLjAiIHk9IjgwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjE2w5cxNiDihpIgNTEyLCBMTjwvdGV4dD48cmVjdCB4PSI1NTAiIHk9IjQwIiB3aWR0aD0iMTMwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNiIgc3Ryb2tlLWRhc2hhcnJheT0iNiA0Ii8+PHRleHQgeD0iNjE1LjAiIHk9IjY0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPnBvc3RfZXh0cmFjdF9wcm9qPC90ZXh0Pjx0ZXh0IHg9IjYxNS4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj41MTIg4oaSIDc2ODwvdGV4dD48cGF0aCBkPSJNMTAwIDY4IEwxMjUgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik0yMzUgNjggTDI2MCA2OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTM3MCA2OCBMMzk1IDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNTI1IDY4IEw1NTAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjx0ZXh0IHg9IjcwNSIgeT0iNjIiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5OID0gOCDDlyAyNCA9IDE5MiB0b2tlbjwvdGV4dD48dGV4dCB4PSI3MDUiIHk9Ijc4IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+dGjhu6kgdOG7sSAodGjhu51pIGdpYW4sIHThuqduIHPhu5EpPC90ZXh0PjxwYXRoIGQ9Ik02MTUgOTYgTDYxNSAxMzIgTDQ4IDEzMiBMNDggMTcwIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSIzMzAiIHk9IjEyNiIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzc0MTUxIj5wb3NfY29udiArIExheWVyTm9ybSwgcuG7k2kgMTIgbOG7m3AgVHJhbnNmb3JtZXIgKHBvc3QtTE4sIERlZXBOb3JtKTwvdGV4dD48cmVjdCB4PSIyMCIgeT0iMTcwIiB3aWR0aD0iNTYiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNDguMCIgeT0iMjAyLjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TDE8L3RleHQ+PHBhdGggZD0iTTc2IDE5OCBMODIgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI4MiIgeT0iMTcwIiB3aWR0aD0iNTYiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMTEwLjAiIHk9IjIwMi41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkwyPC90ZXh0PjxwYXRoIGQ9Ik0xMzggMTk4IEwxNDQgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSIxNDQiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjE3Mi4wIiB5PSIyMDIuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MMzwvdGV4dD48cGF0aCBkPSJNMjAwIDE5OCBMMjA2IDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMjA2IiB5PSIxNzAiIHdpZHRoPSI1NiIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZWY5YzMiIHN0cm9rZT0iI2NhOGEwNCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSIyMzQuMCIgeT0iMjAyLjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TDQ8L3RleHQ+PHBhdGggZD0iTTI2MiAxOTggTDI2OCAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjI2OCIgeT0iMTcwIiB3aWR0aD0iNTYiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMjk2LjAiIHk9IjIwMi41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkw1PC90ZXh0PjxwYXRoIGQ9Ik0zMjQgMTk4IEwzMzAgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSIzMzAiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjM1OC4wIiB5PSIyMDIuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MNjwvdGV4dD48cGF0aCBkPSJNMzg2IDE5OCBMMzkyIDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMzkyIiB5PSIxNzAiIHdpZHRoPSI1NiIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZWY5YzMiIHN0cm9rZT0iI2NhOGEwNCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI0MjAuMCIgeT0iMjAyLjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TDc8L3RleHQ+PHBhdGggZD0iTTQ0OCAxOTggTDQ1NCAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjQ1NCIgeT0iMTcwIiB3aWR0aD0iNTYiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNDgyLjAiIHk9IjIwMi41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkw4PC90ZXh0PjxwYXRoIGQ9Ik01MTAgMTk4IEw1MTYgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI1MTYiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjU0NC4wIiB5PSIyMDIuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MOTwvdGV4dD48cGF0aCBkPSJNNTcyIDE5OCBMNTc4IDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iNTc4IiB5PSIxNzAiIHdpZHRoPSI1NiIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZWY5YzMiIHN0cm9rZT0iI2NhOGEwNCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI2MDYuMCIgeT0iMjAyLjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TDEwPC90ZXh0PjxwYXRoIGQ9Ik02MzQgMTk4IEw2NDAgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI2NDAiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjY2OC4wIiB5PSIyMDIuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MMTE8L3RleHQ+PHBhdGggZD0iTTY5NiAxOTggTDcwMiAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjcwMiIgeT0iMTcwIiB3aWR0aD0iNTYiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNzMwLjAiIHk9IjIwMi41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkwxMjwvdGV4dD48cGF0aCBkPSJNMzkxIDIyNiBMMzkxIDI3MiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMTY2IiB5PSIyNzIiIHdpZHRoPSI0NTAiIGhlaWdodD0iNjIiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMzkxLjAiIHk9IjI5Mi4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPnEsIGssIHYgY+G7p2EgY+G6oyAxMiBs4bubcDwvdGV4dD48dGV4dCB4PSIzOTEuMCIgeT0iMzA3LjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+aG9vayBxX3Byb2ogLyBrX3Byb2ogLyB2X3Byb2osIHRyxrDhu5tjIGtoaSBjaGlhIGhlYWQ8L3RleHQ+PHRleHQgeD0iMzkxLjAiIHk9IjMyMy4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPigxMiwgMywgQiwgMTkyLCA3NjgpPC90ZXh0PjxwYXRoIGQ9Ik03NzYgMTk4IEw4MDAgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI4MDAiIHk9IjE3MCIgd2lkdGg9IjE5MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI4OTUuMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPm1lYW4gcG9vbCB0b2tlbjwvdGV4dD48dGV4dCB4PSI4OTUuMCIgeT0iMjEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmVtYmVkZGluZyAoQiwgNzY4KSDihpIgQ1JEPC90ZXh0PjxwYXRoIGQ9Ik05MTAgMjI2IEw5MTAgMjcwIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI4MzAiIHk9IjI3MCIgd2lkdGg9IjE2MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1kYXNoYXJyYXk9IjYgNCIvPjx0ZXh0IHg9IjkxMC4wIiB5PSIyOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+aGVhZCBMaW5lYXIgNzY44oaSMzY8L3RleHQ+PHRleHQgeD0iOTEwLjAiIHk9IjMxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5sb2dpdCDihpIgTF9sb2dpdDwvdGV4dD48L3N2Zz4="/></p>

    <p align="center"><em>Hình 2. Teacher: lấy q/k/v của cả 12 lớp và embedding cho CRD.</em></p>
    """)
    return


@app.cell
def _(
    BEATS_CONFIG,
    BEAT_FT_PATH,
    FRONTEND,
    Path,
    TARGET_RMS,
    TEACHER_LAYERS,
    Tensor,
    amp,
    device,
    logger,
    namedtuple,
    nn,
    torch,
):
    # (3) Teacher
    from src.model.beat import BEATClassifier
    from src.model.reviewkd import rms_normalize

    TeacherOutput = namedtuple("TeacherOutput", "qkv embedding logits")


    def load_fold_teacher(path: Path) -> BEATClassifier:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        backbone = {k.removeprefix("encoder.model."): v for k, v in state.items() if k.startswith("encoder.model.")}
        beats = BEATClassifier.from_yaml(BEATS_CONFIG)
        converted = beats.convert_checkpoint({"model": backbone})
        converted["head.fc.weight"], converted["head.fc.bias"] = state["head.weight"], state["head.bias"]
        result = beats.load_state_dict(converted, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]}")
        logger.info("teacher fold %s | epoch %s | clean_val_accuracy %.4f | macro_f1 %.4f", checkpoint["fold"],
                    checkpoint["best_epoch"], checkpoint["clean_val_accuracy"], checkpoint["clean_val_macro_f1"])
        return beats


    class QKVTeacher(nn.Module):
        """nhiễu thật -> RMS -> fbank -> BEATs. Trả q/k/v (L, 3, B, N, 768), embedding (B, 768), logits (B, 36)."""

        def __init__(self, beats: BEATClassifier, layers: tuple[int, ...], target_rms: float) -> None:
            super().__init__()
            self.beats = beats.freeze()
            self.layers = tuple(layers)
            self.target_rms = target_rms
            self.freq_bins = beats.frontend.num_mel_bins // beats.config.input_patch_size
            self._captured: dict[tuple[int, str], Tensor] = {}
            for layer in self.layers:
                attention = beats.backbone.encoder.layers[layer - 1].self_attn
                for name in "qkv":
                    getattr(attention, f"{name}_proj").register_forward_hook(self._hook(layer, name))

        def _hook(self, layer: int, name: str):
            def capture(module, inputs, output):
                self._captured[(layer, name)] = output
            return capture

        def train(self, mode: bool = True) -> "QKVTeacher":
            return super().train(False)

        @torch.no_grad()
        def forward(self, noise: Tensor) -> TeacherOutput:
            fbank = self.beats.frontend(rms_normalize(noise.float(), self.target_rms))  # fp32: kaldi fbank
            with amp():
                output = self.beats(fbank, is_waveform=False)
            qkv = torch.stack([torch.stack([self._captured[(layer, name)].float() for name in "qkv"]) for layer in self.layers])
            self._captured.clear()
            return TeacherOutput(qkv, output.features.float().mean(dim=1), output.logits.float())


    teacher = QKVTeacher(load_fold_teacher(BEAT_FT_PATH), TEACHER_LAYERS, TARGET_RMS).to(device)
    with torch.no_grad():
        _probe = teacher(torch.randn(2, 4 * FRONTEND["sample_rate"], device=device) * 0.05)
    TEACHER_TOKENS = _probe.qkv.shape[3]
    GRID = (teacher.freq_bins, TEACHER_TOKENS // teacher.freq_bins)  # (tần số, thời gian) = (8, 24)
    logger.info("teacher qkv %s | embedding %s | lưới token %s", tuple(_probe.qkv.shape), tuple(_probe.embedding.shape), GRID)
    del _probe
    return GRID, teacher


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Student: mixture → FRCRN (mask, đóng băng) → n̂ → log-mel → MobileViT 12 lớp

    `FRCRNSeparator.forward` đã trả n̂ = mixture − ŝ khi `target: speech` trong `frcrn.yaml`. Student là MobileViT-S sửa theo đặc tả: stem, các khối MV2 và ba khối MobileViT với **L = (2, 6, 4) lớp Transformer, d = 768, 12 head × 64, FFN 1536, patch 2×2**, conv 1×1 cuối 768 kênh. Tổng 12 lớp, mỗi lớp khớp một lớp BEATs về d và số head nên không cần adapter: q, k, v lấy thẳng từ `attn.qkv` của từng lớp (trung bình theo 4 vị trí trong patch).

    Đầu ra: trung bình theo trục tần số, rồi attention pooling theo thời gian cho từng lớp (nhánh `att` softmax, nhánh `cla` logit). Dự án này là phân loại **đơn nhãn 36 lớp, clip 4 s (398 khung)**, nên logit đi vào CE; đặc tả gốc là AudioSet 527 lớp đa nhãn, 10 s (998 khung) với BCE và sigmoid, và những phần đó không áp dụng ở đây. FRCRN và log-mel chạy fp32, chỉ MobileViT chạy bf16.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center"><img alt="Student FRCRN và MobileViT 12 lớp" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAwIiBoZWlnaHQ9IjQyMCIgdmlld0JveD0iMCAwIDEwMDAgNDIwIj48dGl0bGU+U3R1ZGVudDogRlJDUk4gdsOgIE1vYmlsZVZpVCAxMiBs4bubcDwvdGl0bGU+PGRlZnM+PG1hcmtlciBpZD0iYXJyIiB2aWV3Qm94PSIwIDAgMTAgMTAiIHJlZlg9IjkiIHJlZlk9IjUiIG1hcmtlcldpZHRoPSI3IiBtYXJrZXJIZWlnaHQ9IjciIG9yaWVudD0iYXV0by1zdGFydC1yZXZlcnNlIj48cGF0aCBkPSJNMCAwIEwxMCA1IEwwIDEwIHoiIGZpbGw9IiM0YjU1NjMiLz48L21hcmtlcj48L2RlZnM+PHJlY3Qgd2lkdGg9IjEwMDAiIGhlaWdodD0iNDIwIiBmaWxsPSIjZmZmZmZmIi8+PHJlY3QgeD0iMTAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjQiLz48dGV4dCB4PSIzMiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5TdHVkZW50PC90ZXh0PjxyZWN0IHg9IjE2MCIgeT0iOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNCIvPjx0ZXh0IHg9IjE4MiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5LaOG7kWkgTW9iaWxlVmlUPC90ZXh0PjxyZWN0IHg9IjMxMCIgeT0iOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNCIvPjx0ZXh0IHg9IjMzMiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5Ew7luZyBjaG8gbG9zczwvdGV4dD48cmVjdCB4PSI0NjAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNmM2Y0ZjYiIHN0cm9rZT0iIzZiNzI4MCIgc3Ryb2tlLXdpZHRoPSIxLjQiIHN0cm9rZS1kYXNoYXJyYXk9IjQgMyIvPjx0ZXh0IHg9IjQ4MiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj7EkMOzbmcgYsSDbmcgLyBk4buvIGxp4buHdTwvdGV4dD48cmVjdCB4PSIxMCIgeT0iNDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM2YjcyODAiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNjAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TWl4dHVyZTwvdGV4dD48dGV4dCB4PSI2MC4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj4oQiwgNjQwMDApPC90ZXh0PjxyZWN0IHg9IjE0MCIgeT0iNDAiIHdpZHRoPSIxOTAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42IiBzdHJva2UtZGFzaGFycmF5PSI2IDQiLz48dGV4dCB4PSIyMzUuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+RlJDUk4gKG1hc2spPC90ZXh0Pjx0ZXh0IHg9IjIzNS4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5TVEZUIOKGkiBVLU5ldCDihpIgbWFzayDihpIgaVNURlQ8L3RleHQ+PHJlY3QgeD0iMzYwIiB5PSI0MCIgd2lkdGg9IjEyMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI0MjAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+bsyCID0gbWl4dHVyZSDiiJIgxZ08L3RleHQ+PHRleHQgeD0iNDIwLjAiIHk9IjgwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCLCA2NDAwMCk8L3RleHQ+PHJlY3QgeD0iNTEwIiB5PSI0MCIgd2lkdGg9IjEyMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI1NzAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+bG9nLW1lbDwvdGV4dD48dGV4dCB4PSI1NzAuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+Y2h14bqpbiBob8OhIHRoZW8gY2xpcDwvdGV4dD48cmVjdCB4PSI2NjAiIHk9IjQwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjcyMC4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5TcGVjQXVnbWVudDwvdGV4dD48dGV4dCB4PSI3MjAuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+Y2jhu4kgbMO6YyB0cmFpbjwvdGV4dD48cmVjdCB4PSI4MTAiIHk9IjQwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2YzZjRmNiIgc3Ryb2tlPSIjNmI3MjgwIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjkwMC4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij4oQiwgMSwgMTI4LCAzOTgpPC90ZXh0Pjx0ZXh0IHg9IjkwMC4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj7EkeG6p3UgdsOgbyBNb2JpbGVWaVQ8L3RleHQ+PHBhdGggZD0iTTExMCA2OCBMMTQwIDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNMzMwIDY4IEwzNjAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik00ODAgNjggTDUxMCA2OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTYzMCA2OCBMNjYwIDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNzgwIDY4IEw4MTAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik05MDAgOTYgTDkwMCAxMzIgTDcwIDEzMiBMNzAgMTcwIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSI1MDAiIHk9IjE1MiIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzc0MTUxIj5Nb2JpbGVWaVQtUyBz4butYTogZCA9IDc2OCwgMTIgaGVhZCDDlyA2NCwgRkZOIDE1MzYsIHBhdGNoIDLDlzIgKDQgduG7iyB0csOtIG3hu5dpIHBhdGNoKTwvdGV4dD48cmVjdCB4PSIyMCIgeT0iMTcwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjgwLjAiIHk9IjE5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5zdGVtICsgTVYyPC90ZXh0Pjx0ZXh0IHg9IjgwLjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj4xNiDihpIgMzIg4oaSIDY0PC90ZXh0PjxwYXRoIGQ9Ik0xNDAgMTk4IEwxNTAgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSIxNTAiIHk9IjE3MCIgd2lkdGg9IjgwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjE5MC4wIiB5PSIxOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TVYyIOKGkzI8L3RleHQ+PHRleHQgeD0iMTkwLjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj45NiDCtyAxNsOXNTA8L3RleHQ+PHBhdGggZD0iTTIzMCAxOTggTDI0MCAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjI0MCIgeT0iMTcwIiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjI5Mi4wIiB5PSIxOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TW9iaWxlVmlUIDE8L3RleHQ+PHRleHQgeD0iMjkyLjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5MID0gMjwvdGV4dD48cGF0aCBkPSJNMzQ0IDE5OCBMMzU0IDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMzU0IiB5PSIxNzAiIHdpZHRoPSI4MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSIzOTQuMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPk1WMiDihpMyPC90ZXh0Pjx0ZXh0IHg9IjM5NC4wIiB5PSIyMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+MTI4IMK3IDjDlzI1PC90ZXh0PjxwYXRoIGQ9Ik00MzQgMTk4IEw0NDQgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI0NDQiIHk9IjE3MCIgd2lkdGg9IjEwNCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZWY5YzMiIHN0cm9rZT0iI2NhOGEwNCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI0OTYuMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPk1vYmlsZVZpVCAyPC90ZXh0Pjx0ZXh0IHg9IjQ5Ni4wIiB5PSIyMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+TCA9IDY8L3RleHQ+PHBhdGggZD0iTTU0OCAxOTggTDU1OCAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjU1OCIgeT0iMTcwIiB3aWR0aD0iODAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNTk4LjAiIHk9IjE5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5NVjIg4oaTMjwvdGV4dD48dGV4dCB4PSI1OTguMCIgeT0iMjEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjE2MCDCtyA0w5cxMzwvdGV4dD48cGF0aCBkPSJNNjM4IDE5OCBMNjQ4IDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iNjQ4IiB5PSIxNzAiIHdpZHRoPSIxMDQiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNzAwLjAiIHk9IjE5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5Nb2JpbGVWaVQgMzwvdGV4dD48dGV4dCB4PSI3MDAuMCIgeT0iMjEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPkwgPSA0PC90ZXh0PjxwYXRoIGQ9Ik03NTIgMTk4IEw3NjIgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI3NjIiIHk9IjE3MCIgd2lkdGg9Ijk2IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjgxMC4wIiB5PSIxOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+Y29udiAxw5cxPC90ZXh0Pjx0ZXh0IHg9IjgxMC4wIiB5PSIyMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+4oaSIDc2OCDCtyA0w5cxMzwvdGV4dD48cGF0aCBkPSJNODU4IDE5OCBMODY4IDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iODY4IiB5PSIxNzAiIHdpZHRoPSI5NiIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI5MTYuMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPmF0dCBwb29saW5nPC90ZXh0Pjx0ZXh0IHg9IjkxNi4wIiB5PSIyMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+bG9naXQgMzYg4oaSIENFPC90ZXh0PjxwYXRoIGQ9Ik0yOTIuMCAyMjYgTDI5Mi4wIDI4MiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMjAwLjAiIHk9IjI4MiIgd2lkdGg9IjE4NCIgaGVpZ2h0PSI3MCIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSIyOTIuMCIgeT0iMzA2LjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+YXR0bi5xa3YgdOG7q25nIGzhu5twPC90ZXh0Pjx0ZXh0IHg9IjI5Mi4wIiB5PSIzMjEuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5s4bubcCAxLTIg4oaUIEJFQVRzIDEtMjwvdGV4dD48dGV4dCB4PSIyOTIuMCIgeT0iMzM3LjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+cSwgaywgdiAoQiwgMjAwLCA3NjgpPC90ZXh0PjxwYXRoIGQ9Ik00OTYuMCAyMjYgTDQ5Ni4wIDI4MiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iNDA0LjAiIHk9IjI4MiIgd2lkdGg9IjE4NCIgaGVpZ2h0PSI3MCIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI0OTYuMCIgeT0iMzA2LjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+YXR0bi5xa3YgdOG7q25nIGzhu5twPC90ZXh0Pjx0ZXh0IHg9IjQ5Ni4wIiB5PSIzMjEuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5s4bubcCAxLTYg4oaUIEJFQVRzIDMtODwvdGV4dD48dGV4dCB4PSI0OTYuMCIgeT0iMzM3LjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+cSwgaywgdiAoQiwgNTIsIDc2OCk8L3RleHQ+PHBhdGggZD0iTTcwMC4wIDIyNiBMNzAwLjAgMjgyIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI2MDguMCIgeT0iMjgyIiB3aWR0aD0iMTg0IiBoZWlnaHQ9IjcwIiByeD0iOCIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjcwMC4wIiB5PSIzMDYuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5hdHRuLnFrdiB04burbmcgbOG7m3A8L3RleHQ+PHRleHQgeD0iNzAwLjAiIHk9IjMyMS41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmzhu5twIDEtNCDihpQgQkVBVHMgOS0xMjwvdGV4dD48dGV4dCB4PSI3MDAuMCIgeT0iMzM3LjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+cSwgaywgdiAoQiwgMTQsIDc2OCk8L3RleHQ+PHBhdGggZD0iTTgxMC4wIDIyNiBMODEwLjAgMjgyIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI3OTYiIHk9IjI4MiIgd2lkdGg9IjEyNCIgaGVpZ2h0PSI3MCIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI4NTguMCIgeT0iMzEzLjc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPm1lYW4gcG9vbDwvdGV4dD48dGV4dCB4PSI4NTguMCIgeT0iMzI5LjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmVtYi4gNzY4IOKGkiBDUkQ8L3RleHQ+PHRleHQgeD0iNTAwIiB5PSIzOTIiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtd2VpZ2h0PSI2MDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiMxMTE4MjciPjEyIGzhu5twIFRyYW5zZm9ybWVyIOKGlCAxMiBs4bubcCBCRUFUcyB0aGVvIHRo4bupIHThu7E7IGzGsOG7m2kgdG9rZW4gdGVhY2hlciAoOMOXMjQpIMSRxrDhu6NjIHBvb2wgduG7gSBsxrDhu5tpIGPhu6dhIHThu6tuZyBraOG7kWkgc3R1ZGVudDwvdGV4dD48L3N2Zz4="/></p>

    <p align="center"><em>Hình 3. Student: FRCRN cho n̂, MobileViT 12 lớp (2 + 6 + 4) và điểm lấy q/k/v của từng lớp.</em></p>
    """)
    return


@app.cell
def _(
    FRCRN_CONFIG,
    FRCRN_PATH,
    FRONTEND,
    GRID,
    NUM_CLASSES,
    SPECAUG,
    STUDENT,
    TEACHER_LAYERS,
    TRAIN,
    Tensor,
    amp,
    device,
    logger,
    math,
    namedtuple,
    nn,
    torch,
):
    # (4) Student: MobileViT 12 lớp, q/k/v của từng lớp lấy bằng hook trên attn.qkv
    from timm.models.byobnet import ByobNet, ByoBlockCfg, ByoModelCfg
    from timm.models.mobilevit import MobileVitBlock

    from src.model.frcrn import FRCRNSeparator
    from src.model.kdmoe.frontend import LogMel, SpecAugment

    StudentOutput = namedtuple("StudentOutput", "logits qkv grids embedding")


    def mobilevit12_cfg(spec: dict) -> ByoModelCfg:
        def inverted(d, c, s):
            return ByoBlockCfg(type="bottle", d=d, c=c, s=s, gs=1, br=4.0, block_kwargs=dict(bottle_in=True, linear_out=True))

        def vit_stage(c, depth):  # MV2 stride 2 rồi khối MobileViT
            return (inverted(1, c, 2), ByoBlockCfg(type="mobilevit", d=1, c=c, s=1, block_kwargs=dict(
                transformer_dim=spec["dim"], transformer_depth=depth, patch_size=spec["patch"], num_heads=spec["heads"],
                mlp_ratio=spec["ffn_dim"] / spec["dim"])))

        blocks = (inverted(1, 32, 1), inverted(3, 64, 2), *[vit_stage(c, d) for c, d in zip(spec["channels"], spec["layers"])])
        return ByoModelCfg(blocks=blocks, stem_chs=16, stem_type="3x3", stem_pool="", downsample="", act_layer="silu",
                           num_features=spec["dim"])


    class FRCRNMobileViT(nn.Module):
        def __init__(self, separator: FRCRNSeparator, spec: dict) -> None:
            super().__init__()
            self.separator = separator.freeze()
            self.frontend = LogMel(**FRONTEND)
            self.spec_augment = SpecAugment(**SPECAUG)
            self.net = ByobNet(mobilevit12_cfg(spec), num_classes=0, in_chans=1)
            self.dropout = nn.Dropout(TRAIN["dropout"])
            self.att = nn.Linear(spec["dim"], NUM_CLASSES)  # attention pooling: softmax theo thời gian, riêng từng lớp
            self.cla = nn.Linear(spec["dim"], NUM_CLASSES)
            self._qkv, self._grids, self._grid, self._batch = [], [], None, 0
            for block in (m for m in self.net.modules() if isinstance(m, MobileVitBlock)):
                block.conv_1x1.register_forward_hook(self._record_grid(spec["patch"]))
                for layer in block.transformer:
                    layer.attn.qkv.register_forward_hook(self._record_qkv)

        def _record_grid(self, patch: int):
            def hook(module, inputs, output):
                self._grid = (math.ceil(output.shape[-2] / patch), math.ceil(output.shape[-1] / patch))  # (tần số, thời gian)
            return hook

        def _record_qkv(self, module, inputs, output) -> None:
            # output (B*P, N, 3D) -> q, k, v (B, N, D), trung bình theo P vị trí trong patch; N theo thứ tự (tần số, thời gian)
            q, k, v = (t.view(self._batch, -1, t.shape[1], t.shape[2]).float().mean(dim=1) for t in output.chunk(3, dim=-1))
            self._qkv.append(torch.stack([q, k, v]))
            self._grids.append(self._grid)

        def train(self, mode: bool = True) -> "FRCRNMobileViT":
            super().train(mode)
            self.separator.eval()
            return self

        def forward(self, waveform: Tensor) -> StudentOutput:
            with torch.no_grad(), torch.autocast(waveform.device.type, enabled=False):
                noise_hat = self.separator(waveform.float())
                x, _ = self.frontend(noise_hat)
            x = self.spec_augment(x)
            self._qkv, self._grids, self._batch = [], [], x.shape[0]
            with amp():
                h = self.net.forward_features(x)                      # (B, 768, H, W)
                z = self.dropout(h.mean(dim=2).transpose(1, 2))      # trung bình theo tần số -> (B, T, 768)
                weight = self.att(z).float().softmax(dim=1)          # softmax theo thời gian
                logits = (weight * self.cla(z).float()).sum(dim=1)   # (B, 36)
            return StudentOutput(logits, tuple(self._qkv), tuple(self._grids), h.float().mean(dim=(2, 3)))

        def predict_proba(self, waveform: Tensor) -> Tensor:
            return self(waveform).logits.softmax(dim=-1)


    def build_student() -> FRCRNMobileViT:
        separator = FRCRNSeparator.from_yaml(FRCRN_CONFIG)
        report = separator.load_pretrained(FRCRN_PATH)
        logger.info("FRCRN %s | missing %d unexpected %d", FRCRN_PATH.name, len(report.missing), len(report.unexpected))
        return FRCRNMobileViT(separator, STUDENT)


    student = build_student().to(device)
    with torch.no_grad():
        _probe = student.eval()(torch.randn(2, 4 * FRONTEND["sample_rate"], device=device) * 0.05)
    EMBED_CHANNELS = _probe.embedding.shape[1]
    assert len(_probe.qkv) == len(TEACHER_LAYERS), f"student có {len(_probe.qkv)} lớp, teacher {len(TEACHER_LAYERS)}"
    logger.info("student %.2fM tham số (transformer %.2fM) | embedding %d", sum(p.numel() for p in student.net.parameters()) / 1e6,
                sum(p.numel() for n, p in student.net.named_parameters() if ".transformer." in n) / 1e6, EMBED_CHANNELS)
    _owner = [block for block, depth in enumerate(STUDENT["layers"], 1) for _ in range(depth)]
    for _index, (_qkv, _grid) in enumerate(zip(_probe.qkv, _probe.grids)):
        logger.info("student lớp %2d (khối %d) q/k/v %s lưới token %s <-> BEATs lớp %2d (lưới %s)", _index + 1, _owner[_index],
                    tuple(_qkv.shape), _grid, TEACHER_LAYERS[_index], GRID)
    del _probe
    return EMBED_CHANNELS, FRCRNMobileViT, build_student


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss KD: q/k/v và CRD
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center"><img alt="Loss q/k/v theo từng lớp" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAwIiBoZWlnaHQ9IjM4MCIgdmlld0JveD0iMCAwIDEwMDAgMzgwIj48dGl0bGU+TG9zcyBxLCBrLCB2IHRoZW8gdOG7q25nIGzhu5twPC90aXRsZT48ZGVmcz48bWFya2VyIGlkPSJhcnIiIHZpZXdCb3g9IjAgMCAxMCAxMCIgcmVmWD0iOSIgcmVmWT0iNSIgbWFya2VyV2lkdGg9IjciIG1hcmtlckhlaWdodD0iNyIgb3JpZW50PSJhdXRvLXN0YXJ0LXJldmVyc2UiPjxwYXRoIGQ9Ik0wIDAgTDEwIDUgTDAgMTAgeiIgZmlsbD0iIzRiNTU2MyIvPjwvbWFya2VyPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzODAiIGZpbGw9IiNmZmZmZmYiLz48cmVjdCB4PSIxMCIgeT0iOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNCIvPjx0ZXh0IHg9IjMyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPlN0dWRlbnQ8L3RleHQ+PHJlY3QgeD0iMTYwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iMTgyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPlRlYWNoZXI8L3RleHQ+PHJlY3QgeD0iMzEwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iMzMyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPkxvc3M8L3RleHQ+PHJlY3QgeD0iMTAiIHk9IjMwIiB3aWR0aD0iMTQwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjgwLjAiIHk9IjU0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkzhu5twIFRyYW5zZm9ybWVyIGk8L3RleHQ+PHRleHQgeD0iODAuMCIgeT0iNzAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+a2jhu5FpIGIgY+G7p2Egc3R1ZGVudDwvdGV4dD48cmVjdCB4PSIxNzUiIHk9IjMwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjI2NS4wIiB5PSI1NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5hdHRuLnFrdiAoaG9vayk8L3RleHQ+PHRleHQgeD0iMjY1LjAiIHk9IjcwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCwrc0LCBOX3MsIDMgw5cgNzY4KTwvdGV4dD48cmVjdCB4PSIzODAiIHk9IjMwIiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjQ2MC4wIiB5PSI1NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5jaGlhIHEsIGssIHY8L3RleHQ+PHRleHQgeD0iNDYwLjAiIHk9IjcwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPlRCIHRoZW8gNCB24buLIHRyw608L3RleHQ+PHJlY3QgeD0iNTY1IiB5PSIzMCIgd2lkdGg9IjIzMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI2ODAuMCIgeT0iNTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+ccyCLCBrzIIsIHbMgiBj4bunYSBzdHVkZW50PC90ZXh0Pjx0ZXh0IHg9IjY4MC4wIiB5PSI3MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj4oQiwgTl9zLCA3NjgpLCBOX3MgdGhlbyBraOG7kWk8L3RleHQ+PHBhdGggZD0iTTE1MCA1OCBMMTc1IDU4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNMzU1IDU4IEwzODAgNTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik01NDAgNTggTDU2NSA1OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMTAiIHk9IjI3MCIgd2lkdGg9IjE0MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkYmVhZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1kYXNoYXJyYXk9IjYgNCIvPjx0ZXh0IHg9IjgwLjAiIHk9IjI5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5CRUFUcyBs4bubcCBpJzwvdGV4dD48dGV4dCB4PSI4MC4wIiB5PSIzMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+KMSRw7NuZyBixINuZyk8L3RleHQ+PHJlY3QgeD0iMTc1IiB5PSIyNzAiIHdpZHRoPSIxODAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMjY1LjAiIHk9IjI5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5xL2svdl9wcm9qIChob29rKTwvdGV4dD48dGV4dCB4PSIyNjUuMCIgeT0iMzEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCLCAxOTIsIDc2OCkgbeG7l2kgbG/huqFpPC90ZXh0PjxyZWN0IHg9IjM4MCIgeT0iMjcwIiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjQ2MC4wIiB5PSIyOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+cG9vbCB0cnVuZyBiw6xuaDwvdGV4dD48dGV4dCB4PSI0NjAuMCIgeT0iMzEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjjDlzI0IOKGkiBsxrDhu5tpIGto4buRaTwvdGV4dD48cmVjdCB4PSI1NjUiIHk9IjI3MCIgd2lkdGg9IjIzMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkYmVhZmUiIHN0cm9rZT0iIzI1NjNlYiIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI2ODAuMCIgeT0iMjk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPnEsIGssIHYgY+G7p2EgdGVhY2hlcjwvdGV4dD48dGV4dCB4PSI2ODAuMCIgeT0iMzEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCLCBOX3MsIDc2OCk8L3RleHQ+PHBhdGggZD0iTTE1MCAyOTggTDE3NSAyOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik0zNTUgMjk4IEwzODAgMjk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNTQwIDI5OCBMNTY1IDI5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iNTAwIiB5PSIxNTAiIHdpZHRoPSIxNzAiIGhlaWdodD0iNjIiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNTg1LjAiIHk9IjE3MC4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkxfY29zPC90ZXh0Pjx0ZXh0IHg9IjU4NS4wIiB5PSIxODUuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5tZWFuKDEg4oiSIGNvcyh4zIIsIHgpKTwvdGV4dD48dGV4dCB4PSI1ODUuMCIgeT0iMjAxLjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+eCDiiIgge3EsIGssIHZ9LCB04burbmcgdG9rZW48L3RleHQ+PHJlY3QgeD0iNzIwIiB5PSIxNTAiIHdpZHRoPSIyMjAiIGhlaWdodD0iNjIiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iODMwLjAiIHk9IjE3MC4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkxfcmVsPC90ZXh0Pjx0ZXh0IHg9IjgzMC4wIiB5PSIxODUuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5LTChSX3Qg4oCWIFJfcyk8L3RleHQ+PHRleHQgeD0iODMwLjAiIHk9IjIwMS4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPlIgPSBzb2Z0bWF4KHhfaCB4X2jhtYAv4oiaNjQpLCAxMiBoZWFkPC90ZXh0PjxwYXRoIGQ9Ik02MTAgODYgTDYxMCAxNTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik03NjAgODYgTDc2MCAxNTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik02MTAgMjcwIEw2MTAgMjEyIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNzYwIDI3MCBMNzYwIDIxMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHRleHQgeD0iMjAiIHk9IjE1MCIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9IjYwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj7DgW5oIHjhuqEgaSDihpQgaScgKGPDuW5nIHRo4bupIHThu7EpOjwvdGV4dD48dGV4dCB4PSIyMCIgeT0iMTY4IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+a2jhu5FpIDE6IGzhu5twIDEtMiDihpQgQkVBVHMgMS0yIChsxrDhu5tpIDjDlzI1KTwvdGV4dD48dGV4dCB4PSIyMCIgeT0iMTg0IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+a2jhu5FpIDI6IGzhu5twIDEtNiDihpQgQkVBVHMgMy04IChsxrDhu5tpIDTDlzEzKTwvdGV4dD48dGV4dCB4PSIyMCIgeT0iMjAwIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+a2jhu5FpIDM6IGzhu5twIDEtNCDihpQgQkVBVHMgOS0xMiAobMaw4bubaSAyw5c3KTwvdGV4dD48dGV4dCB4PSIyMCIgeT0iMjI0IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIHRleHQtYW5jaG9yPSJzdGFydCIgZmlsbD0iIzM3NDE1MSI+Q8O5bmcgZCA9IDc2OCB2w6AgMTIgaGVhZCBuw6puIGtow7RuZyBj4bqnbiBhZGFwdGVyLjwvdGV4dD48dGV4dCB4PSI1MDAiIHk9IjM2MCIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzc0MTUxIj5MX3FrdiA9IHRydW5nIGLDrG5oIHF1YSAxMiBs4bubcCBj4bunYSAoTF9jb3MgKyBSRUxfV0VJR0hUIMK3IExfcmVsKS4gQ29zaW5lIGtow7RuZyBwaOG7pSB0aHXhu5ljIHRoYW5nIMSRbzsgS0wgw6lwIGPhuqV1IHRyw7pjIGF0dGVudGlvbiBnaeG7kW5nIHRlYWNoZXIuPC90ZXh0Pjwvc3ZnPg=="/></p>

    <p align="center"><em>Hình 4. q/k/v của từng lớp student so với lớp BEATs tương ứng bằng cosine và KL quan hệ (không cần adapter).</em></p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center"><img alt="CRD" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5NjAiIGhlaWdodD0iNDAwIiB2aWV3Qm94PSIwIDAgOTYwIDQwMCI+PHRpdGxlPkNSRCB24bubaSBtZW1vcnkgYmFuazwvdGl0bGU+PGRlZnM+PG1hcmtlciBpZD0iYXJyIiB2aWV3Qm94PSIwIDAgMTAgMTAiIHJlZlg9IjkiIHJlZlk9IjUiIG1hcmtlcldpZHRoPSI3IiBtYXJrZXJIZWlnaHQ9IjciIG9yaWVudD0iYXV0by1zdGFydC1yZXZlcnNlIj48cGF0aCBkPSJNMCAwIEwxMCA1IEwwIDEwIHoiIGZpbGw9IiM0YjU1NjMiLz48L21hcmtlcj48L2RlZnM+PHJlY3Qgd2lkdGg9Ijk2MCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiNmZmZmZmYiLz48cmVjdCB4PSIxMCIgeT0iOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNCIvPjx0ZXh0IHg9IjMyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPlN0dWRlbnQ8L3RleHQ+PHJlY3QgeD0iMTYwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iMTgyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPlRlYWNoZXI8L3RleHQ+PHJlY3QgeD0iMzEwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iMzMyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPkxvc3M8L3RleHQ+PHJlY3QgeD0iNDYwIiB5PSI4IiB3aWR0aD0iMTYiIGhlaWdodD0iMTIiIHJ4PSIzIiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM2YjcyODAiIHN0cm9rZS13aWR0aD0iMS40Ii8+PHRleHQgeD0iNDgyIiB5PSIxOC41IiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ic3RhcnQiIGZpbGw9IiMzNzQxNTEiPk1lbW9yeSBiYW5rPC90ZXh0PjxyZWN0IHg9IjEwIiB5PSI0MCIgd2lkdGg9IjExMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI2NS4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5FbWJlZGRpbmc8L3RleHQ+PHRleHQgeD0iNjUuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+c3R1ZGVudCAoQiwgNjQwKTwvdGV4dD48cmVjdCB4PSIxNTAiIHk9IjQwIiB3aWR0aD0iMTEwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjIwNS4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MaW5lYXIgNjQw4oaSMTI4PC90ZXh0Pjx0ZXh0IHg9IjIwNS4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5zdHVkZW50IGhlYWQ8L3RleHQ+PHJlY3QgeD0iMjkwIiB5PSI0MCIgd2lkdGg9IjExMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSIzNDUuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TDItbm9ybWFsaXplPC90ZXh0Pjx0ZXh0IHg9IjM0NS4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj48L3RleHQ+PHJlY3QgeD0iNDMwIiB5PSI0MCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjQ2MC4wIiB5PSI3Mi41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPnpfczwvdGV4dD48cmVjdCB4PSIxMCIgeT0iMjYwIiB3aWR0aD0iMTEwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjY1LjAiIHk9IjI4NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5FbWJlZGRpbmc8L3RleHQ+PHRleHQgeD0iNjUuMCIgeT0iMzAwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPnRlYWNoZXIgKEIsIDc2OCk8L3RleHQ+PHJlY3QgeD0iMTUwIiB5PSIyNjAiIHdpZHRoPSIxMTAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMjA1LjAiIHk9IjI4NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5MaW5lYXIgNzY44oaSMTI4PC90ZXh0Pjx0ZXh0IHg9IjIwNS4wIiB5PSIzMDAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+dGVhY2hlciBoZWFkPC90ZXh0PjxyZWN0IHg9IjI5MCIgeT0iMjYwIiB3aWR0aD0iMTEwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RiZWFmZSIgc3Ryb2tlPSIjMjU2M2ViIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjM0NS4wIiB5PSIyODQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TDItbm9ybWFsaXplPC90ZXh0Pjx0ZXh0IHg9IjM0NS4wIiB5PSIzMDAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+PC90ZXh0PjxyZWN0IHg9IjQzMCIgeT0iMjYwIiB3aWR0aD0iNjAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGJlYWZlIiBzdHJva2U9IiMyNTYzZWIiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNDYwLjAiIHk9IjI5Mi41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPnpfdDwvdGV4dD48cGF0aCBkPSJNMTIwIDY4IEwxNTAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik0yNjAgNjggTDI5MCA2OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTQwMCA2OCBMNDMwIDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNMTIwIDI4OCBMMTUwIDI4OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTI2MCAyODggTDI5MCAyODgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik00MDAgMjg4IEw0MzAgMjg4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI1NjAiIHk9IjExMCIgd2lkdGg9IjE3MCIgaGVpZ2h0PSI1MCIgcng9IjgiIGZpbGw9IiNmM2Y0ZjYiIHN0cm9rZT0iIzZiNzI4MCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI2NDUuMCIgeT0iMTMxLjc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPmJhbmtfdGVhY2hlcjwvdGV4dD48dGV4dCB4PSI2NDUuMCIgeT0iMTQ3LjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPk4gw5cgMTI4PC90ZXh0PjxyZWN0IHg9IjU2MCIgeT0iMjAwIiB3aWR0aD0iMTcwIiBoZWlnaHQ9IjUwIiByeD0iOCIgZmlsbD0iI2YzZjRmNiIgc3Ryb2tlPSIjNmI3MjgwIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjY0NS4wIiB5PSIyMjEuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+YmFua19zdHVkZW50PC90ZXh0Pjx0ZXh0IHg9IjY0NS4wIiB5PSIyMzcuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+TiDDlyAxMjg8L3RleHQ+PHJlY3QgeD0iNzgwIiB5PSI0MCIgd2lkdGg9IjE3MCIgaGVpZ2h0PSI3NiIgcng9IjgiIGZpbGw9IiNmZmVkZDUiIHN0cm9rZT0iI2VhNTgwYyIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI4NjUuMCIgeT0iNjcuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5JbmZvTkNFIChhbmNob3Igel9zKTwvdGV4dD48dGV4dCB4PSI4NjUuMCIgeT0iODIuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5kxrDGoW5nOiB6X3QgY8O5bmcgbeG6q3U8L3RleHQ+PHRleHQgeD0iODY1LjAiIHk9Ijk4LjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+w6JtOiBLPTQwOTYgdOG7qyBiYW5rX3RlYWNoZXI8L3RleHQ+PHJlY3QgeD0iNzgwIiB5PSIyNDAiIHdpZHRoPSIxNzAiIGhlaWdodD0iNzYiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iODY1LjAiIHk9IjI2Ny4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPkluZm9OQ0UgKGFuY2hvciB6X3QpPC90ZXh0Pjx0ZXh0IHg9Ijg2NS4wIiB5PSIyODIuNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5kxrDGoW5nOiB6X3MgY8O5bmcgbeG6q3U8L3RleHQ+PHRleHQgeD0iODY1LjAiIHk9IjI5OC4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPsOibTogSz00MDk2IHThu6sgYmFua19zdHVkZW50PC90ZXh0PjxwYXRoIGQ9Ik00OTAgNjggTDc4MCA2OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTQ5MCAyODggTDc4MCAyODgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik03MzAgMTI1IEw3NTUgMTI1IEw3NTUgMTAwIEw3ODAgMTAwIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNzMwIDIyNSBMNzU1IDIyNSBMNzU1IDI2MCBMNzgwIDI2MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTQ0NSA5NiBMNDQ1IDIyNSBMNTYwIDIyNSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgc3Ryb2tlLWRhc2hhcnJheT0iNSA0IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNDc1IDI2MCBMNDc1IDEzNSBMNTYwIDEzNSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgc3Ryb2tlLWRhc2hhcnJheT0iNSA0IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSI1MDAiIHk9IjE4NCIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5j4bqtcCBuaOG6rXQgYmFuayAobW9tZW50dW0gMC41KTwvdGV4dD48dGV4dCB4PSI0ODAiIHk9IjM3MiIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzc0MTUxIj5MX2NyZCA9IEluZm9OQ0Uoel9zKSArIEluZm9OQ0Uoel90KSwgz4QgPSAwLjA3LiBOZWdhdGl2ZSBjw7luZyBs4bubcCBob+G6t2MgY8O5bmcgY2jhu4kgc+G7kSBt4bqrdSBi4buLIG1hc2sg4oiS4oieICh0csOhbmggZmFsc2UgbmVnYXRpdmUpLjwvdGV4dD48L3N2Zz4="/></p>

    <p align="center"><em>Hình 5. CRD: hai head chiếu, memory bank hai view và InfoNCE đối xứng.</em></p>
    """)
    return


@app.cell
def _(F, Tensor, math, nn, torch):
    # (5) Pool q/k/v teacher về lưới student, loss q/k/v, loss CRD
    def pool_teacher(qkv: Tensor, grid_teacher: tuple[int, int], grid_student: tuple[int, int]) -> Tensor:
        """(3, B, N_t, D), token BEATs theo thứ tự (thời gian, tần số) -> (3, B, N_s, D) trên lưới student, thứ tự (tần số, thời gian)."""
        three, batch, tokens, dim = qkv.shape
        freq, time = grid_teacher
        x = qkv.reshape(three * batch, time, freq, dim).permute(0, 3, 2, 1)  # (3B, D, F, T)
        if (freq, time) != tuple(grid_student):
            x = F.adaptive_avg_pool2d(x, grid_student)
        return x.permute(0, 2, 3, 1).reshape(three, batch, -1, dim)


    def relation(x: Tensor, heads: int) -> Tensor:
        """(B, N, D) -> softmax(x_h x_h^T / sqrt(d)) theo từng head, (B*H*N, N)."""
        batch, tokens, dim = x.shape
        x = x.view(batch, tokens, heads, dim // heads).transpose(1, 2)
        return (x @ x.transpose(-2, -1) / math.sqrt(dim // heads)).reshape(-1, tokens)


    class QKVLoss(nn.Module):
        def __init__(self, heads: int = 12, rel_weight: float = 1.0) -> None:
            super().__init__()
            self.heads, self.rel_weight = heads, rel_weight

        def forward(self, student_qkv: Tensor, teacher_qkv: Tensor) -> tuple[Tensor, Tensor]:
            """Cả hai (3, B, N, D). Trả (cosine, KL quan hệ), mỗi cái trung bình trên q, k, v."""
            cosine, kl = student_qkv.new_zeros(()), student_qkv.new_zeros(())
            for s, t in zip(student_qkv, teacher_qkv):
                cosine = cosine + (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
                if self.rel_weight > 0:
                    kl = kl + F.kl_div(F.log_softmax(relation(s, self.heads), dim=-1),
                                       F.softmax(relation(t, self.heads), dim=-1), reduction="batchmean")
            return cosine / 3, kl / 3


    class CRDLoss(nn.Module):
        """Contrastive Representation Distillation với memory bank cho hai view, đối xứng teacher <-> student."""

        def __init__(self, student_dim: int, teacher_dim: int, num_samples: int, embed_dim: int, negatives: int,
                     temperature: float, momentum: float) -> None:
            super().__init__()
            self.embed_student = nn.Linear(student_dim, embed_dim)
            self.embed_teacher = nn.Linear(teacher_dim, embed_dim)
            self.negatives, self.temperature, self.momentum = negatives, temperature, momentum
            self.register_buffer("bank_student", F.normalize(torch.randn(num_samples, embed_dim), dim=1))
            self.register_buffer("bank_teacher", F.normalize(torch.randn(num_samples, embed_dim), dim=1))
            self.register_buffer("bank_labels", torch.full((num_samples,), -1, dtype=torch.long))

        def _nce(self, anchor: Tensor, positive: Tensor, bank: Tensor, negative_index: Tensor, index: Tensor, labels: Tensor) -> Tensor:
            positive_logit = (anchor * positive).sum(dim=1, keepdim=True)
            negative_logit = torch.einsum("be,bke->bk", anchor, bank[negative_index])
            invalid = (self.bank_labels[negative_index] == labels[:, None]) | (negative_index == index[:, None])
            negative_logit = negative_logit.masked_fill(invalid, float("-inf"))
            logits = torch.cat([positive_logit, negative_logit], dim=1) / self.temperature
            return F.cross_entropy(logits, torch.zeros(len(anchor), dtype=torch.long, device=anchor.device))

        def forward(self, student_embedding: Tensor, teacher_embedding: Tensor, index: Tensor, labels: Tensor) -> Tensor:
            zs = F.normalize(self.embed_student(student_embedding), dim=1)
            zt = F.normalize(self.embed_teacher(teacher_embedding), dim=1)
            negative_index = torch.randint(0, self.bank_student.shape[0], (len(index), self.negatives), device=index.device)
            loss = (self._nce(zs, zt, self.bank_teacher, negative_index, index, labels)
                    + self._nce(zt, zs, self.bank_student, negative_index, index, labels))
            with torch.no_grad():
                for bank, z in ((self.bank_student, zs), (self.bank_teacher, zt)):
                    bank[index] = F.normalize(self.momentum * bank[index] + (1.0 - self.momentum) * z.detach(), dim=1)
                self.bank_labels[index] = labels
            return loss

    return CRDLoss, QKVLoss, pool_teacher


@app.cell
def _(Path, ROOT, tqdm):
    # (5b) Tải dataset mixture-36l từ Kaggle (nếu chưa có) và kiểm tra đủ tệp. Token đọc từ KAGGLE_ACCESS_TOKEN, không ghi vào notebook.
    import csv
    import os
    import time

    from scripts import download_dataset as dl
    from scripts.load_dataset import MIXTURE_KEY

    DOWNLOAD_ROOT = ROOT / "datasets"


    def ensure_mixture36(download_root: Path) -> Path:
        print(f"[1/3] tìm {MIXTURE_KEY} (env, url.yaml), nếu thiếu sẽ tải từ Kaggle vào {download_root}")
        if not os.environ.get(dl.TOKEN_ENV):
            print(f"      chưa đặt {dl.TOKEN_ENV}: chỉ dùng được dataset đã có trên máy")
        started = time.perf_counter()
        root = dl.ensure_dataset(MIXTURE_KEY, download_root)  # thanh tqdm: byte khi tải, tệp khi giải nén
        print(f"[2/3] dataset tại {root} ({time.perf_counter() - started:.0f}s)")
        print("[3/3] kiểm tra tệp mixture và noise theo manifest")
        for split in ("train", "validation", "test"):
            with open(root / split / "manifest.csv", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            missing = sum(not (root / row[key]).is_file() for row in tqdm(rows, desc=split, unit="mẫu") for key in ("mixture_path", "noise_path"))
            print(f"      {split}: {len(rows)} mẫu, thiếu {missing} tệp")
            if missing:
                raise FileNotFoundError(f"{split}: thiếu {missing} tệp, xoá thư mục {root} rồi tải lại")
        return root


    DATASET_ROOT = ensure_mixture36(DOWNLOAD_ROOT)
    return (DATASET_ROOT,)


@app.cell
def _(Dataset, torch):
    # (6) Dữ liệu: train có nhiễu thật và chỉ số mẫu (cho memory bank CRD)
    from scripts.finetune import SMOKE_BATCH, SMOKE_SAMPLES
    from scripts.load_dataset import Mixture36Dataset, build_dataloader, subsample

    DURATION = 4.0


    class IndexedDataset(Dataset):
        def __init__(self, base: Dataset) -> None:
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> dict:
            item = dict(self.base[index])
            item["index"] = torch.tensor(index)
            return item


    def with_fork(loader):
        """Kernel marimo mặc định tạo worker bằng spawn, không pickle được IndexedDataset (định nghĩa trong cell). Worker chỉ đọc tệp, không dùng CUDA, nên fork an toàn."""
        if loader.num_workers > 0:
            loader.multiprocessing_context = "fork"
        return loader


    def build_loaders(train_config: dict, max_samples: int | None, workers: int) -> dict:
        train_ds = IndexedDataset(subsample(Mixture36Dataset(split="train", duration=DURATION, load_noise=True), max_samples))
        val_ds = subsample(Mixture36Dataset(split="validation", duration=DURATION), max_samples)
        return {
            "train": with_fork(build_dataloader(train_ds, train_config["batch_size"], True, workers)),
            "validation": with_fork(build_dataloader(val_ds, train_config["eval_batch_size"], False, workers)),
            "num_train": len(train_ds),
        }

    return (
        DURATION,
        Mixture36Dataset,
        SMOKE_BATCH,
        SMOKE_SAMPLES,
        build_dataloader,
        build_loaders,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kiểm tra teacher (tuỳ chọn, rẻ)

    Teacher phải phân loại nhiễu thật đúng cỡ `clean_val_accuracy` trong checkpoint (fold 0: 0.74). Nếu thấp hơn nhiều thì đường vào (RMS, fbank) lệch so với lúc fine-tune, nên dừng lại trước khi huấn luyện student.
    """)
    return


@app.cell
def _(
    DURATION,
    Mixture36Dataset,
    build_dataloader,
    device,
    logger,
    teacher,
    torch,
):
    # (7) Độ chính xác của teacher trên nhiễu thật của tập validation
    @torch.no_grad()
    def check_teacher(max_batches: int = 20, batch_size: int = 32) -> float:
        loader = build_dataloader(Mixture36Dataset(split="validation", duration=DURATION, load_noise=True), batch_size, False, 0)
        correct = seen = 0
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            output = teacher(batch["noise"].to(device))
            correct += (output.logits.argmax(1).cpu() == batch["target"].argmax(1)).sum().item()
            seen += len(batch["target"])
        logger.info("teacher: accuracy trên nhiễu thật (validation, %d mẫu) = %.4f", seen, correct / seen)
        return correct / seen


    # check_teacher()   # bỏ comment để chạy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Huấn luyện và đánh giá
    """)
    return


@app.cell
def _(
    AdamW,
    CRDLoss,
    DRIVE_DIR,
    DURATION,
    EMBED_CHANNELS,
    F,
    FRCRNMobileViT,
    FRCRN_PATH,
    GRID,
    KD,
    Mixture36Dataset,
    NUM_CLASSES,
    NUM_WORKERS,
    QKVLoss,
    ROOT,
    RUN_TRAINING,
    SMOKE_BATCH,
    SMOKE_SAMPLES,
    SNR_LEVELS,
    STUDENT,
    TEACHER_LAYERS,
    TRAIN,
    USE_CRD,
    USE_LOGIT,
    USE_QKV,
    build_dataloader,
    build_loaders,
    build_student,
    copy,
    device,
    fs,
    logger,
    nn,
    pd,
    pool_teacher,
    run,
    set_seed,
    teacher,
    torch,
    tqdm,
):
    # (8) Vòng huấn luyện + đánh giá 9 chỉ số theo từng SNR
    from src.metrics import ClassificationMetrics  # noqa: F401  (snr_scores dùng bên dưới)
    from src.train.beat_trainer import cosine_with_warmup
    from src.train.reviewkd_trainer import NINE_METRICS, snr_scores
    from scripts.drive_sync import DriveCheckpointUploader, DriveFileBackup, DriveLogUploader
    from src.utils.checkpoint import save_checkpoint


    @torch.no_grad()
    def evaluate(model: FRCRNMobileViT, loader, desc: str = "evaluate") -> dict[str, float]:
        model.eval()
        probabilities, labels, snrs = [], [], []
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            probabilities.append(model.predict_proba(batch["waveform"].to(device)).float().cpu())
            labels.append(batch["target"].argmax(dim=1))
            snrs.append(batch["snr_db"])
        return snr_scores(torch.cat(probabilities).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), NUM_CLASSES)


    def per_snr_table(scores: dict[str, float]) -> pd.DataFrame:
        rows = {f"{level}": [scores.get(f"{name}_snr_{level}", float("nan")) for name in NINE_METRICS] for level in SNR_LEVELS}
        rows["mean"] = [scores.get(f"{name}_snr_mean", float("nan")) for name in NINE_METRICS]
        return pd.DataFrame.from_dict(rows, orient="index", columns=NINE_METRICS).rename_axis("snr_db")


    def kd_scale(epoch: int, warmup: int) -> float:
        return 1.0 if warmup == 0 else min(1.0, (epoch - 1) / warmup)


    def run_experiment(name: str) -> tuple[dict, pd.DataFrame | None]:
        train_config, kd = dict(TRAIN), dict(KD)
        max_samples, workers = None, NUM_WORKERS
        if not RUN_TRAINING:
            train_config.update(epochs=1, patience=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
            kd["warmup_epochs"] = 0
            max_samples, workers = SMOKE_SAMPLES, 0

        set_seed(train_config["seed"])
        loaders = build_loaders(train_config, max_samples, workers)
        model = build_student().to(device)
        qkv_loss = QKVLoss(heads=12, rel_weight=kd["rel_weight"]).to(device)
        crd = (CRDLoss(EMBED_CHANNELS, 768, loaders["num_train"], kd["crd_dim"], kd["crd_negatives"],
                       kd["crd_temperature"], kd["crd_momentum"]).to(device) if USE_CRD else None)

        trainable = [p for module in (model, crd) if module is not None for p in module.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=train_config["lr"], weight_decay=train_config["weight_decay"], betas=train_config["betas"])
        steps = train_config["epochs"] * len(loaders["train"])
        scheduler = cosine_with_warmup(optimizer, steps, train_config["warmup_ratio"], train_config["min_lr_ratio"])
        logger.info("%s | student %.2fM | crd %.2fM | train %d mẫu", name,
                    sum(p.numel() for p in model.net.parameters()) / 1e6,
                    sum(p.numel() for p in crd.parameters()) / 1e6 if crd else 0.0, loaders["num_train"])

        history, best_score, best_epoch, best_state, stale, start_epoch = [], -float("inf"), 0, None, 0, 1
        flags = dict(qkv=USE_QKV, crd=USE_CRD, logit=USE_LOGIT, layers=TEACHER_LAYERS, student=STUDENT, kd=kd)
        resume_path = ROOT / "output" / "resume" / name / "resume.pt" if RUN_TRAINING else None  # thư mục cố định để Run all nối tiếp được
        remote_dir = f"{DRIVE_DIR}/{name}"
        use_drive = fs is not None and RUN_TRAINING
        drive_best = DriveCheckpointUploader(fs, remote_dir, logger) if use_drive else None
        drive_resume = DriveFileBackup(fs, remote_dir, "resume.pt", logger) if use_drive else None
        drive_history = DriveFileBackup(fs, remote_dir, f"history_{name}.csv", logger) if use_drive else None
        drive_log = DriveLogUploader(fs, run.log_file, remote_dir) if use_drive else None
        if drive_log is not None:
            logger.addHandler(drive_log)
        if use_drive and not resume_path.is_file():  # local mất (sandbox mới) thì lấy resume.pt từ Drive
            try:
                if fs.exists(f"{remote_dir}/resume.pt"):
                    resume_path.parent.mkdir(parents=True, exist_ok=True)
                    fs.get_file(f"{remote_dir}/resume.pt", str(resume_path))
                    logger.info("drive: khôi phục resume.pt từ %s", remote_dir)
            except Exception as error:
                logger.warning("drive: không khôi phục được resume.pt: %r", error)
        if resume_path is not None and resume_path.is_file():
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            if state["flags"] != flags:
                raise ValueError(f"{resume_path}: cấu hình khác lần chạy trước ({state['flags']} != {flags}), xoá file để chạy lại từ đầu")
            model.net.load_state_dict(state["net"])
            if crd is not None:
                crd.load_state_dict(state["crd"])  # gồm cả memory bank
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, best_score, best_epoch, best_state, stale = state["history"], state["best_score"], state["best_epoch"], state["best_state"], state["stale"]
            start_epoch = state["epoch"] + 1
            set_seed(train_config["seed"] + start_epoch)
            logger.info("%s: nối tiếp từ epoch %d (best %.4f ở epoch %d, stale %d)", name, start_epoch, best_score, best_epoch, stale)
        for epoch in range(start_epoch, train_config["epochs"] + 1):
            model.train()
            scale = kd_scale(epoch, kd["warmup_epochs"])
            sums = dict(loss=0.0, ce=0.0, cos=0.0, rel=0.0, crd=0.0, kl=0.0, correct=0, seen=0)
            cos_layer_sum = None  # 1 - cos(q,k,v) của từng lớp, để thấy lớp nào học chậm
            for batch in tqdm(loaders["train"], desc=f"epoch {epoch}", unit="batch", leave=False, dynamic_ncols=True):
                waveform, noise = batch["waveform"].to(device), batch["noise"].to(device)
                labels, index = batch["target"].argmax(dim=1).to(device), batch["index"].to(device)
                output = model(waveform)
                ce = F.cross_entropy(output.logits, labels, label_smoothing=train_config["label_smoothing"])
                loss, cos, rel, nce, kl = ce, ce.new_zeros(()), ce.new_zeros(()), ce.new_zeros(()), ce.new_zeros(())
                if USE_QKV or USE_CRD or USE_LOGIT:
                    teacher_output = teacher(noise)
                if USE_QKV:
                    pairs = [qkv_loss(output.qkv[i], pool_teacher(teacher_output.qkv[i], GRID, output.grids[i])) for i in range(len(output.qkv))]
                    cos_layers, rel_layers = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
                    cos, rel = cos_layers.mean(), rel_layers.mean()
                    step_layers = cos_layers.detach() * len(labels)
                    cos_layer_sum = step_layers if cos_layer_sum is None else cos_layer_sum + step_layers
                    loss = loss + scale * kd["qkv_weight"] * (cos + kd["rel_weight"] * rel)
                if USE_CRD:
                    nce = crd(output.embedding, teacher_output.embedding, index, labels)
                    loss = loss + scale * kd["crd_weight"] * nce
                if USE_LOGIT:
                    temperature = kd["logit_temperature"]
                    kl = F.kl_div(F.log_softmax(output.logits / temperature, dim=-1),
                                  F.softmax(teacher_output.logits / temperature, dim=-1), reduction="batchmean") * temperature**2
                    loss = loss + scale * kd["logit_weight"] * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, train_config["grad_clip"])
                optimizer.step()
                scheduler.step()
                n = len(labels)
                for key, value in (("loss", loss), ("ce", ce), ("cos", cos), ("rel", rel), ("crd", nce), ("kl", kl)):
                    sums[key] += value.item() * n
                sums["correct"] += (output.logits.argmax(1) == labels).sum().item()
                sums["seen"] += n
            seen = max(sums["seen"], 1)
            scores = evaluate(model, loaders["validation"], desc=f"epoch {epoch} - validation")
            record = {"epoch": epoch, "kd_scale": scale, "train_acc": sums["correct"] / seen,
                      **{f"train_{k}": sums[k] / seen for k in ("loss", "ce", "cos", "rel", "crd", "kl")},
                      **({f"train_cos_l{i + 1}": v / seen for i, v in enumerate(cos_layer_sum.tolist())} if cos_layer_sum is not None else {}),
                      **{f"val_{k}": v for k, v in scores.items()}}
            history.append(record)
            logger.info("%s epoch %d | loss %.4f ce %.4f cos %.4f rel %.4f crd %.4f kl %.4f | train acc %.4f | val macro_acc %.4f mAP_macro %.4f",
                        name, epoch, record["train_loss"], record["train_ce"], record["train_cos"], record["train_rel"],
                        record["train_crd"], record["train_kl"], record["train_acc"], scores["macro_acc"], scores["mAP_macro"])
            if RUN_TRAINING:
                run.result_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(run.result_dir / f"history_{name}.csv", index=False)
                if drive_history is not None:
                    drive_history(run.result_dir / f"history_{name}.csv")
            if scores["macro_acc"] > best_score:
                best_score, best_epoch, stale = scores["macro_acc"], epoch, 0
                best_state = copy.deepcopy(model.net.state_dict())
                if RUN_TRAINING:  # lưu ngay khi tốt hơn, để sandbox mất kết nối giữa chừng vẫn còn checkpoint
                    save_checkpoint(run.checkpoint_path(name), model.net, epoch=epoch, macro_acc=best_score, frcrn=str(FRCRN_PATH.relative_to(ROOT)))
                    if drive_best is not None:
                        drive_best(run.checkpoint_path(name), epoch, best_score)  # ghi đè best.pt trên Drive
            else:
                stale += 1
            if resume_path is not None:  # ghi nguyên tử, để bị ngắt giữa lúc ghi vẫn còn bản cũ
                resume_path.parent.mkdir(parents=True, exist_ok=True)
                partial = resume_path.with_name(resume_path.name + ".part")
                torch.save({"epoch": epoch, "flags": flags, "net": model.net.state_dict(),
                            "crd": crd.state_dict() if crd is not None else None,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
                            "best_score": best_score, "best_epoch": best_epoch, "best_state": best_state, "stale": stale}, partial)
                partial.replace(resume_path)
                if drive_resume is not None:
                    drive_resume(resume_path)
            if stale >= train_config["patience"]:
                logger.info("%s: dừng sớm ở epoch %d (best %d, %.4f)", name, epoch, best_epoch, best_score)
                break

        model.net.load_state_dict(best_state)
        val_table = per_snr_table({k.removeprefix("val_"): v for k, v in history[best_epoch - 1].items() if k.startswith("val_")})
        logger.info("%s: validation tại epoch tốt nhất %d, 9 chỉ số theo SNR\n%s", name, best_epoch, val_table.round(4).to_string())
        test_table = None
        if RUN_TRAINING:
            test_loader = build_dataloader(Mixture36Dataset(split="test", duration=DURATION), train_config["eval_batch_size"], False, workers)
            test_table = per_snr_table(evaluate(model, test_loader, desc=f"{name} - test"))
            logger.info("%s: TEST, 9 chỉ số theo SNR\n%s", name, test_table.round(4).to_string())
            checkpoint_path = run.checkpoint_path(name)
            save_checkpoint(checkpoint_path, model.net, epoch=best_epoch, macro_acc=best_score, frcrn=str(FRCRN_PATH.relative_to(ROOT)))
            run.result_dir.mkdir(parents=True, exist_ok=True)
            val_table.to_csv(run.result_dir / f"per_snr_validation_{name}.csv")
            test_table.to_csv(run.result_dir / f"per_snr_test_{name}.csv")
            if use_drive:
                for table_file in (f"per_snr_validation_{name}.csv", f"per_snr_test_{name}.csv"):
                    DriveFileBackup(fs, remote_dir, table_file, logger)(run.result_dir / table_file)
            pd.DataFrame(history).to_csv(run.result_dir / f"history_{name}.csv", index=False)
        if drive_log is not None:  # đẩy log lần cuối rồi gỡ handler để chạy lại không bị nhân đôi
            drive_log.upload()
            logger.removeHandler(drive_log)
            drive_log.close()
        return {"model": name, "best_epoch": best_epoch, "val_macro_acc": best_score}, (test_table if test_table is not None else val_table)

    return (run_experiment,)


@app.cell
def _(
    DATASET_ROOT,
    RUN_TRAINING,
    USE_CRD,
    USE_LOGIT,
    USE_QKV,
    device,
    run_experiment,
):
    # (9) Chạy một thí nghiệm theo các cờ ở cell (1). RUN_TRAINING = True huấn luyện đầy đủ, False chạy smoke test nhỏ.
    # Chỉ chạy cell này khi đã đồng ý dùng máy.
    name = "-".join(["mobilevit12", "frcrn", *(["qkv"] if USE_QKV else []), *(["crd"] if USE_CRD else []), *(["logit"] if USE_LOGIT else []),
                      *([] if (USE_QKV or USE_CRD or USE_LOGIT) else ["ce"])])
    print(f"dataset: {DATASET_ROOT} | thí nghiệm: {name} | RUN_TRAINING={RUN_TRAINING} | thiết bị: {device}")
    summary, table = run_experiment(name)
    print(summary)
    table.round(4)
    return


if __name__ == "__main__":
    app.run()
