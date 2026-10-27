# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "timm==1.0.30",
#     "tqdm==4.70.1",
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

    **Loss** = CE (label smoothing 0.1) + λ_qkv · L_qkv + λ_crd · L_crd + λ_logit · L_logit + λ_hn · L_hn. Ba λ của KD tăng tuyến tính trong `warmup_epochs` epoch đầu (hoặc giảm tuyến tính về 0 ở nhánh ESKD). L_hn là hard-negative theo nhóm lớp dễ nhầm, **không dùng teacher** và không nhân hệ số KD nên độc lập với ablation KD; mặc định tắt (λ_hn = 0).

    - **L_qkv** (mỗi cặp lớp, mỗi loại q, k, v): `1 − cos` theo từng token + `REL_WEIGHT` · KL giữa ma trận quan hệ `softmax(xxᵀ/√d)` theo từng head (kiểu MiniLM). Cosine không phụ thuộc thang đo, nên q, k, v có độ lớn khác nhau vẫn so sánh được. Student có cùng d = 768 và 12 head với BEATs nên so q, k, v trực tiếp, không cần adapter. Student lấy q/k/v từ `attn.qkv` của từng lớp (trung bình theo 4 vị trí trong patch); q/k/v của teacher được pool trung bình về lưới token của khối student (8×25, 4×13, 2×7 với clip 4 s). Có thể giới hạn trong một tập lớp BEATs bằng `qkv_layers` (ví dụ 9-12) mà không đổi ánh xạ lớp.
    - **L_crd** (Tian et al., 2020): hai head chiếu về 128 chiều, L2-norm. Positive là cùng mẫu ở hai view, negative lấy `CRD_NEGATIVES` mẫu ngẫu nhiên từ memory bank (cập nhật momentum). Negative cùng lớp với anchor bị loại, để tránh false negative vì tập chỉ có 36 lớp.
    - **L_logit** (Hinton et al., 2015): `T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))` với T = 4 (có thể đổi bằng khoá `T`). Teacher nhìn nhiễu sạch nên logit của nó là nhãn mềm "có đặc quyền"; đổi lại teacher fold 0 chỉ đạt khoảng 0.74 accuracy trên nhiễu sạch và 0.66 trên n̂ của FRCRN (validation), nên nhãn mềm có nhiễu và CE vẫn giữ vai trò neo.
    - **Đánh giá:** 9 chỉ số (top1, top3, balanced acc, precision, recall, macro-F1, micro-F1, mAP, macro-AUC) tại từng mức SNR (−5…20) và trung bình. Checkpoint chọn theo `macro_acc` (balanced acc trung bình theo SNR), đúng chuẩn repo.

    `RUN_TRAINING = False` chỉ chạy smoke test nhỏ (64 mẫu, 1 epoch) vào `output/test/`. Cell (1) có hàng đợi `RUNS`: mỗi nhánh bật/tắt từng loss (`qkv`, `crd`, `logit`), đổi nhiệt độ `T`, giới hạn lớp q/k/v, bật hard-negative `hn`, hoặc khởi tạo từ checkpoint tốt nhất rồi giảm KD (ESKD); xem mục "Huấn luyện và đánh giá". `USE_NHAT_CACHE` tính trước n̂ của FRCRN một lần để epoch nhanh hơn nhiều.

    **Hậu xử lý (sau huấn luyện).** Hiệu chỉnh bias theo lớp `z'_c = z_c + b_c` (`b_c = τ·log(q_c/p_c)`), fit trên val bằng lưới τ, khoá rồi đánh giá test một lần; không đổi trọng số. Chi tiết ở mục "Hiệu chỉnh bias theo lớp" cuối notebook.
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
    from torch.utils.data import Dataset, Subset
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
        Subset,
        Tensor,
        math,
        namedtuple,
        nn,
        np,
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
    USE_LOGIT = True              # L_logit: KL(softmax(z_t/T) || softmax(z_s/T)) trên logit 36 lớp của teacher fold 0 (T = 4, trọng số 1,0 trong KD); False để tắt
    USE_DRIVE = True              # backup best.pt, resume.pt, history, log lên Google Drive (đăng nhập ở cell 2b)
    DRIVE_DIR = "FA26_Capstone/Truong/beat-mv-kd-v3"  # thư mục mới: beat-mv-kd và beat-mv-kd-v2 trên Drive đã có bản trùng tên (MultipleFilesError)
    USE_NHAT_CACHE = True         # tính trước n^ = FRCRN(mixture) một lần (FRCRN đóng băng, mixture không augment nên n^ không đổi); chỉ dùng khi RUN_TRAINING
    NHAT_CACHE_DIR = ROOT / "cache" / "nhat"
    NHAT_ZIP = ROOT / "nhat_cache.zip"                     # nén lossless thư mục cache/ (khoảng 8,6 GB), nằm NGOÀI cache/ để không tự nén chính nó
    NHAT_DRIVE_ZIP = f"{DRIVE_DIR}/nhat/nhat_cache.zip"    # bản lưu trên Drive: sandbox mới chỉ cần tải về và giải nén thay vì tính lại (khoảng 17 phút)
    UPLOAD_NHAT_TO_DRIVE = True                            # đẩy zip lên Drive một lần nếu Drive chưa có (khoảng 8 phút, chặn huấn luyện); False để bỏ qua
    USE_AUG = True                # augmentation nhất quán trên cặp (n^ cho student, nhiễu thật cho teacher): dịch vòng thời gian + mixup; cần USE_NHAT_CACHE
    AUG = dict(shift=True, mixup_prob=0.5, mixup_alpha=0.4)  # mixup_prob: xác suất một batch được trộn; alpha của Beta(alpha, alpha)
    NUM_WORKERS = 8

    BEAT_FOLD = 0
    BEAT_FT_PATH = ROOT / "pretrained" / "beat-ft" / f"beats_teacher_fold{BEAT_FOLD}_weights.pt"
    FRCRN_PATH = ROOT / "pretrained" / "FRCRN" / "best_e037_20261020T203958Z.pt"
    BEATS_CONFIG = ROOT / "config" / "beat.yaml"
    FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"

    TEACHER_LAYERS = tuple(range(1, 13))  # cả 12 lớp BEATs (đếm từ 1) lấy q/k/v
    # MobileViT 12 lớp: khối 1 (2 lớp) <-> BEATs 1-2, khối 2 (6 lớp) <-> BEATs 3-8, khối 3 (4 lớp) <-> BEATs 9-12
    STUDENT = dict(dim=768, heads=12, ffn_dim=768, layers=(2, 6, 4), channels=(96, 128, 160), patch=2)
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

    TRAIN = dict(lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999), warmup_ratio=0.05,
                 lr_factor=0.5, lr_patience=2, lr_threshold=0.005, min_lr=1e-6,  # ReduceLROnPlateau theo val macro_acc, mỗi epoch
                 label_smoothing=0.1, grad_clip=1.0, batch_size=32, eval_batch_size=64,
                 epochs=50, patience=15, dropout=0.3, drop_path=0.1, seed=42)

    # Hard-negative: các nhóm lớp hay nhầm nhau, lấy từ ma trận nhầm lẫn trên validation (epoch 27 của thí nghiệm tham chiếu), nên mức cải thiện trên các lớp này
    # một phần là do chọn nhóm theo validation. Mỗi lớp thuộc nhiều nhất một nhóm; tên khớp đầy đủ hoặc tiền tố với tên lớp của dataset.
    HN_GROUPS = (
        ("Saxophone", "Clarinet", "Trumpet", "Flute", "Harmonica", "Violin"),
        ("Glockenspiel", "Chime", "Electric piano", "Acoustic guitar", "Telephone"),
        ("Knock", "Gunshot", "Shatter", "Fireworks"),
        ("Scissors", "Tearing", "Writing", "Keys jangling", "Computer keyboard"),
        ("Microwave oven", "Gong", "Bus"),
        ("Fart", "Meow"),
    )

    # Hàng đợi thí nghiệm cho cell (9), chạy lần lượt. Mỗi nhánh ablation khác thí nghiệm THAM CHIẾU (cả ba loss KD, 50 epoch, cfg rỗng) đúng một núm,
    # trừ "ce" (bỏ cả ba loss). Thí nghiệm tham chiếu không chạy lại: lấy từ history trong resume.pt của nó để so ở cùng ngân sách epoch.
    # Khoá của một phần tử: tag, qkv, crd, logit (bật/tắt loss), T (nhiệt độ logit), qkv_layers (lớp BEATs 1-12 có loss q/k/v), hn (trọng số hard-negative),
    # init_from ("reference" hoặc đường dẫn resume.pt: khởi tạo từ best_state), kd_decay_epochs (giảm tuyến tính KD về 0), lr, warmup_ratio,
    # max_epochs (dừng sớm nhưng GIỮ lịch lr và warm-up của TRAIN["epochs"], nên so cặp được với tham chiếu), eval_test (False: chưa động tới test).
    # Thời gian ước tính trên GPU sandbox: khoảng 3 phút mỗi epoch có teacher (gồm upload Drive). Hàng đợi mặc định: 4 x 15 = 60 epoch, khoảng 3 giờ.
    RUN_TAG = "r6"                # gắn vào tên mọi thí nghiệm khác tham chiếu, để không đụng resume.pt và thư mục Drive cũ
    RUNS = [
        # ESKD (tuỳ chọn, mặc định tắt): nhánh duy nhất không huấn luyện từ đầu, vì khởi tạo từ best_state của thí nghiệm tham chiếu.
        # dict(tag="eskd", init_from="reference", crd=False, kd_decay_epochs=8, lr=1e-5, warmup_ratio=0.0, max_epochs=10, eval_test=False),  # ESKD: tắt dần KD từ best_state
        dict(tag="no-crd", crd=False, max_epochs=15, eval_test=False),
        dict(tag="no-qkv", qkv=False, max_epochs=15, eval_test=False),
        dict(tag="ce", qkv=False, crd=False, logit=False, max_epochs=15, eval_test=False),
        dict(tag="hn", hn=1.0, max_epochs=15, eval_test=False),
        # Thêm nhánh (mỗi dòng khoảng 40 phút):
        # dict(tag="T2", T=2.0, max_epochs=15, eval_test=False),
        # dict(tag="qkv9-12", qkv_layers=(9, 10, 11, 12), max_epochs=15, eval_test=False),
    ]
    return (
        AUG,
        BEATS_CONFIG,
        BEAT_FT_PATH,
        DRIVE_DIR,
        EXPERIMENT,
        FRCRN_CONFIG,
        FRCRN_PATH,
        FRONTEND,
        HN_GROUPS,
        KD,
        NHAT_CACHE_DIR,
        NHAT_DRIVE_ZIP,
        NHAT_ZIP,
        NUM_CLASSES,
        NUM_WORKERS,
        RUNS,
        RUN_TAG,
        RUN_TRAINING,
        SNR_LEVELS,
        SPECAUG,
        STUDENT,
        TARGET_RMS,
        TEACHER_LAYERS,
        TRAIN,
        UPLOAD_NHAT_TO_DRIVE,
        USE_AUG,
        USE_CRD,
        USE_DRIVE,
        USE_LOGIT,
        USE_NHAT_CACHE,
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

    `FRCRNSeparator.forward` đã trả n̂ = mixture − ŝ khi `target: speech` trong `frcrn.yaml`. Student là MobileViT-S sửa theo đặc tả: stem, các khối MV2 và ba khối MobileViT với **L = (2, 6, 4) lớp Transformer, d = 768, 12 head × 64, FFN 768, patch 2×2**, conv 1×1 cuối 768 kênh. Tổng 12 lớp, mỗi lớp khớp một lớp BEATs về d và số head nên không cần adapter: q, k, v lấy thẳng từ `attn.qkv` của từng lớp (trung bình theo 4 vị trí trong patch).

    Đầu ra: trung bình theo trục tần số, rồi attention pooling theo thời gian cho từng lớp (nhánh `att` softmax, nhánh `cla` logit). Dự án này là phân loại **đơn nhãn 36 lớp, clip 4 s (398 khung)**, nên logit đi vào CE; đặc tả gốc là AudioSet 527 lớp đa nhãn, 10 s (998 khung) với BCE và sigmoid, và những phần đó không áp dụng ở đây. FRCRN và log-mel chạy fp32, chỉ MobileViT chạy bf16.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center"><img alt="Student FRCRN và MobileViT 12 lớp" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAwIiBoZWlnaHQ9IjQyMCIgdmlld0JveD0iMCAwIDEwMDAgNDIwIj48dGl0bGU+U3R1ZGVudDogRlJDUk4gdsOgIE1vYmlsZVZpVCAxMiBs4bubcDwvdGl0bGU+PGRlZnM+PG1hcmtlciBpZD0iYXJyIiB2aWV3Qm94PSIwIDAgMTAgMTAiIHJlZlg9IjkiIHJlZlk9IjUiIG1hcmtlcldpZHRoPSI3IiBtYXJrZXJIZWlnaHQ9IjciIG9yaWVudD0iYXV0by1zdGFydC1yZXZlcnNlIj48cGF0aCBkPSJNMCAwIEwxMCA1IEwwIDEwIHoiIGZpbGw9IiM0YjU1NjMiLz48L21hcmtlcj48L2RlZnM+PHJlY3Qgd2lkdGg9IjEwMDAiIGhlaWdodD0iNDIwIiBmaWxsPSIjZmZmZmZmIi8+PHJlY3QgeD0iMTAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjQiLz48dGV4dCB4PSIzMiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5TdHVkZW50PC90ZXh0PjxyZWN0IHg9IjE2MCIgeT0iOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNCIvPjx0ZXh0IHg9IjE4MiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5LaOG7kWkgTW9iaWxlVmlUPC90ZXh0PjxyZWN0IHg9IjMxMCIgeT0iOCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjEyIiByeD0iMyIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNCIvPjx0ZXh0IHg9IjMzMiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj5Ew7luZyBjaG8gbG9zczwvdGV4dD48cmVjdCB4PSI0NjAiIHk9IjgiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxMiIgcng9IjMiIGZpbGw9IiNmM2Y0ZjYiIHN0cm9rZT0iIzZiNzI4MCIgc3Ryb2tlLXdpZHRoPSIxLjQiIHN0cm9rZS1kYXNoYXJyYXk9IjQgMyIvPjx0ZXh0IHg9IjQ4MiIgeT0iMTguNSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9IjQwMCIgdGV4dC1hbmNob3I9InN0YXJ0IiBmaWxsPSIjMzc0MTUxIj7EkMOzbmcgYsSDbmcgLyBk4buvIGxp4buHdTwvdGV4dD48cmVjdCB4PSIxMCIgeT0iNDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZjNmNGY2IiBzdHJva2U9IiM2YjcyODAiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNjAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TWl4dHVyZTwvdGV4dD48dGV4dCB4PSI2MC4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj4oQiwgNjQwMDApPC90ZXh0PjxyZWN0IHg9IjE0MCIgeT0iNDAiIHdpZHRoPSIxOTAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42IiBzdHJva2UtZGFzaGFycmF5PSI2IDQiLz48dGV4dCB4PSIyMzUuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+RlJDUk4gKG1hc2spPC90ZXh0Pjx0ZXh0IHg9IjIzNS4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5TVEZUIOKGkiBVLU5ldCDihpIgbWFzayDihpIgaVNURlQ8L3RleHQ+PHJlY3QgeD0iMzYwIiB5PSI0MCIgd2lkdGg9IjEyMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI0MjAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+bsyCID0gbWl4dHVyZSDiiJIgxZ08L3RleHQ+PHRleHQgeD0iNDIwLjAiIHk9IjgwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPihCLCA2NDAwMCk8L3RleHQ+PHJlY3QgeD0iNTEwIiB5PSI0MCIgd2lkdGg9IjEyMCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI1NzAuMCIgeT0iNjQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+bG9nLW1lbDwvdGV4dD48dGV4dCB4PSI1NzAuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+Y2h14bqpbiBob8OhIHRoZW8gY2xpcDwvdGV4dD48cmVjdCB4PSI2NjAiIHk9IjQwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjcyMC4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5TcGVjQXVnbWVudDwvdGV4dD48dGV4dCB4PSI3MjAuMCIgeT0iODAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+Y2jhu4kgbMO6YyB0cmFpbjwvdGV4dD48cmVjdCB4PSI4MTAiIHk9IjQwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2YzZjRmNiIgc3Ryb2tlPSIjNmI3MjgwIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjkwMC4wIiB5PSI2NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij4oQiwgMSwgMTI4LCAzOTgpPC90ZXh0Pjx0ZXh0IHg9IjkwMC4wIiB5PSI4MC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj7EkeG6p3UgdsOgbyBNb2JpbGVWaVQ8L3RleHQ+PHBhdGggZD0iTTExMCA2OCBMMTQwIDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNMzMwIDY4IEwzNjAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik00ODAgNjggTDUxMCA2OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHBhdGggZD0iTTYzMCA2OCBMNjYwIDY4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cGF0aCBkPSJNNzgwIDY4IEw4MTAgNjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxwYXRoIGQ9Ik05MDAgOTYgTDkwMCAxMzIgTDcwIDEzMiBMNzAgMTcwIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48dGV4dCB4PSI1MDAiIHk9IjE1MiIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjMzc0MTUxIj5Nb2JpbGVWaVQtUyBz4butYTogZCA9IDc2OCwgMTIgaGVhZCDDlyA2NCwgRkZOIDc2OCwgcGF0Y2ggMsOXMiAoNCB24buLIHRyw60gbeG7l2kgcGF0Y2gpPC90ZXh0PjxyZWN0IHg9IjIwIiB5PSIxNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iODAuMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPnN0ZW0gKyBNVjI8L3RleHQ+PHRleHQgeD0iODAuMCIgeT0iMjEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjE2IOKGkiAzMiDihpIgNjQ8L3RleHQ+PHBhdGggZD0iTTE0MCAxOTggTDE1MCAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjE1MCIgeT0iMTcwIiB3aWR0aD0iODAiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMTkwLjAiIHk9IjE5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5NVjIg4oaTMjwvdGV4dD48dGV4dCB4PSIxOTAuMCIgeT0iMjEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPjk2IMK3IDE2w5c1MDwvdGV4dD48cGF0aCBkPSJNMjMwIDE5OCBMMjQwIDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iMjQwIiB5PSIxNzAiIHdpZHRoPSIxMDQiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZmVmOWMzIiBzdHJva2U9IiNjYThhMDQiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iMjkyLjAiIHk9IjE5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5Nb2JpbGVWaVQgMTwvdGV4dD48dGV4dCB4PSIyOTIuMCIgeT0iMjEwLjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPkwgPSAyPC90ZXh0PjxwYXRoIGQ9Ik0zNDQgMTk4IEwzNTQgMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSIzNTQiIHk9IjE3MCIgd2lkdGg9IjgwIiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2RjZmNlNyIgc3Ryb2tlPSIjMTZhMzRhIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjM5NC4wIiB5PSIxOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TVYyIOKGkzI8L3RleHQ+PHRleHQgeD0iMzk0LjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj4xMjggwrcgOMOXMjU8L3RleHQ+PHBhdGggZD0iTTQzNCAxOTggTDQ0NCAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjQ0NCIgeT0iMTcwIiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZlZjljMyIgc3Ryb2tlPSIjY2E4YTA0IiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjQ5Ni4wIiB5PSIxOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+TW9iaWxlVmlUIDI8L3RleHQ+PHRleHQgeD0iNDk2LjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5MID0gNjwvdGV4dD48cGF0aCBkPSJNNTQ4IDE5OCBMNTU4IDE5OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGI1NTYzIiBzdHJva2Utd2lkdGg9IjEuNiIgbWFya2VyLWVuZD0idXJsKCNhcnIpIi8+PHJlY3QgeD0iNTU4IiB5PSIxNzAiIHdpZHRoPSI4MCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNkY2ZjZTciIHN0cm9rZT0iIzE2YTM0YSIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI1OTguMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPk1WMiDihpMyPC90ZXh0Pjx0ZXh0IHg9IjU5OC4wIiB5PSIyMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+MTYwIMK3IDTDlzEzPC90ZXh0PjxwYXRoIGQ9Ik02MzggMTk4IEw2NDggMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI2NDgiIHk9IjE3MCIgd2lkdGg9IjEwNCIgaGVpZ2h0PSI1NiIgcng9IjgiIGZpbGw9IiNmZWY5YzMiIHN0cm9rZT0iI2NhOGEwNCIgc3Ryb2tlLXdpZHRoPSIxLjYiLz48dGV4dCB4PSI3MDAuMCIgeT0iMTk0Ljc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPk1vYmlsZVZpVCAzPC90ZXh0Pjx0ZXh0IHg9IjcwMC4wIiB5PSIyMTAuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+TCA9IDQ8L3RleHQ+PHBhdGggZD0iTTc1MiAxOTggTDc2MiAxOTgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9Ijc2MiIgeT0iMTcwIiB3aWR0aD0iOTYiIGhlaWdodD0iNTYiIHJ4PSI4IiBmaWxsPSIjZGNmY2U3IiBzdHJva2U9IiMxNmEzNGEiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iODEwLjAiIHk9IjE5NC43NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5jb252IDHDlzE8L3RleHQ+PHRleHQgeD0iODEwLjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj7ihpIgNzY4IMK3IDTDlzEzPC90ZXh0PjxwYXRoIGQ9Ik04NTggMTk4IEw4NjggMTk4IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI4NjgiIHk9IjE3MCIgd2lkdGg9Ijk2IiBoZWlnaHQ9IjU2IiByeD0iOCIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjkxNi4wIiB5PSIxOTQuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+YXR0IHBvb2xpbmc8L3RleHQ+PHRleHQgeD0iOTE2LjAiIHk9IjIxMC4yNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5sb2dpdCAzNiDihpIgQ0U8L3RleHQ+PHBhdGggZD0iTTI5Mi4wIDIyNiBMMjkyLjAgMjgyIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSIyMDAuMCIgeT0iMjgyIiB3aWR0aD0iMTg0IiBoZWlnaHQ9IjcwIiByeD0iOCIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjI5Mi4wIiB5PSIzMDYuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5hdHRuLnFrdiB04burbmcgbOG7m3A8L3RleHQ+PHRleHQgeD0iMjkyLjAiIHk9IjMyMS41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmzhu5twIDEtMiDihpQgQkVBVHMgMS0yPC90ZXh0Pjx0ZXh0IHg9IjI5Mi4wIiB5PSIzMzcuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5xLCBrLCB2IChCLCAyMDAsIDc2OCk8L3RleHQ+PHBhdGggZD0iTTQ5Ni4wIDIyNiBMNDk2LjAgMjgyIiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMS42IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiLz48cmVjdCB4PSI0MDQuMCIgeT0iMjgyIiB3aWR0aD0iMTg0IiBoZWlnaHQ9IjcwIiByeD0iOCIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9IjQ5Ni4wIiB5PSIzMDYuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMi41IiBmb250LXdlaWdodD0iNjAwIiBmaWxsPSIjMTExODI3Ij5hdHRuLnFrdiB04burbmcgbOG7m3A8L3RleHQ+PHRleHQgeD0iNDk2LjAiIHk9IjMyMS41IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjExLjUiIGZvbnQtd2VpZ2h0PSI0MDAiIGZpbGw9IiMzNzQxNTEiPmzhu5twIDEtNiDihpQgQkVBVHMgMy04PC90ZXh0Pjx0ZXh0IHg9IjQ5Ni4wIiB5PSIzMzcuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5xLCBrLCB2IChCLCA1MiwgNzY4KTwvdGV4dD48cGF0aCBkPSJNNzAwLjAgMjI2IEw3MDAuMCAyODIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9IjYwOC4wIiB5PSIyODIiIHdpZHRoPSIxODQiIGhlaWdodD0iNzAiIHJ4PSI4IiBmaWxsPSIjZmZlZGQ1IiBzdHJva2U9IiNlYTU4MGMiIHN0cm9rZS13aWR0aD0iMS42Ii8+PHRleHQgeD0iNzAwLjAiIHk9IjMwNi4wIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iU2Vnb2UgVUksIEFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjEyLjUiIGZvbnQtd2VpZ2h0PSI2MDAiIGZpbGw9IiMxMTE4MjciPmF0dG4ucWt2IHThu6tuZyBs4bubcDwvdGV4dD48dGV4dCB4PSI3MDAuMCIgeT0iMzIxLjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+bOG7m3AgMS00IOKGlCBCRUFUcyA5LTEyPC90ZXh0Pjx0ZXh0IHg9IjcwMC4wIiB5PSIzMzcuMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMS41IiBmb250LXdlaWdodD0iNDAwIiBmaWxsPSIjMzc0MTUxIj5xLCBrLCB2IChCLCAxNCwgNzY4KTwvdGV4dD48cGF0aCBkPSJNODEwLjAgMjI2IEw4MTAuMCAyODIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRiNTU2MyIgc3Ryb2tlLXdpZHRoPSIxLjYiIG1hcmtlci1lbmQ9InVybCgjYXJyKSIvPjxyZWN0IHg9Ijc5NiIgeT0iMjgyIiB3aWR0aD0iMTI0IiBoZWlnaHQ9IjcwIiByeD0iOCIgZmlsbD0iI2ZmZWRkNSIgc3Ryb2tlPSIjZWE1ODBjIiBzdHJva2Utd2lkdGg9IjEuNiIvPjx0ZXh0IHg9Ijg1OC4wIiB5PSIzMTMuNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIuNSIgZm9udC13ZWlnaHQ9IjYwMCIgZmlsbD0iIzExMTgyNyI+bWVhbiBwb29sPC90ZXh0Pjx0ZXh0IHg9Ijg1OC4wIiB5PSIzMjkuMjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJTZWdvZSBVSSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEuNSIgZm9udC13ZWlnaHQ9IjQwMCIgZmlsbD0iIzM3NDE1MSI+ZW1iLiA3Njgg4oaSIENSRDwvdGV4dD48dGV4dCB4PSI1MDAiIHk9IjM5MiIgZm9udC1mYW1pbHk9IlNlZ29lIFVJLCBBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9IjYwMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzExMTgyNyI+MTIgbOG7m3AgVHJhbnNmb3JtZXIg4oaUIDEyIGzhu5twIEJFQVRzIHRoZW8gdGjhu6kgdOG7sTsgbMaw4bubaSB0b2tlbiB0ZWFjaGVyICg4w5cyNCkgxJHGsOG7o2MgcG9vbCB24buBIGzGsOG7m2kgY+G7p2EgdOG7q25nIGto4buRaSBzdHVkZW50PC90ZXh0Pjwvc3ZnPg=="/></p>

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
            self.net = ByobNet(mobilevit12_cfg(spec), num_classes=0, in_chans=1, drop_path_rate=TRAIN["drop_path"])
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

        def forward(self, waveform: Tensor | None = None, noise_hat: Tensor | None = None) -> StudentOutput:
            source = noise_hat if noise_hat is not None else waveform
            with torch.no_grad(), torch.autocast(source.device.type, enabled=False):
                if noise_hat is None:  # không có cache: chạy FRCRN trực tiếp
                    noise_hat = self.separator(waveform.float())
                x, _ = self.frontend(noise_hat.float())
            x = self.spec_augment(x)
            self._qkv, self._grids, self._batch = [], [], x.shape[0]
            with amp():
                h = self.net.forward_features(x)                      # (B, 768, H, W)
                z = self.dropout(h.mean(dim=2).transpose(1, 2))      # trung bình theo tần số -> (B, T, 768)
                weight = self.att(z).float().softmax(dim=1)          # softmax theo thời gian
                logits = (weight * self.cla(z).float()).sum(dim=1)   # (B, 36)
            return StudentOutput(logits, tuple(self._qkv), tuple(self._grids), h.float().mean(dim=(2, 3)))

        def predict_proba(self, waveform: Tensor | None = None, noise_hat: Tensor | None = None) -> Tensor:
            return self(waveform, noise_hat).logits.softmax(dim=-1)


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
    return EMBED_CHANNELS, FRCRNMobileViT, build_student, student


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


    def resolve_groups(groups, class_names: list[str]) -> list[Tensor]:
        """Tên ngắn -> chỉ số lớp. Khớp tên đầy đủ trước, rồi tiền tố (không phân biệt hoa thường); thiếu, mơ hồ hoặc trùng giữa hai nhóm thì báo lỗi."""
        resolved, used = [], set()
        for group in groups:
            indices = []
            for short in group:
                key = short.lower()
                hits = [i for i, c in enumerate(class_names) if c.lower() == key] or [i for i, c in enumerate(class_names) if c.lower().startswith(key)]
                if len(hits) != 1 or hits[0] in used:
                    raise ValueError(f"hard-negative: '{short}' khớp {[class_names[i] for i in hits]} (đã dùng: {hits[0] in used if hits else False})")
                used.add(hits[0])
                indices.append(hits[0])
            resolved.append(torch.tensor(indices, dtype=torch.long))
        return resolved


    def hard_negative_loss(logits: Tensor, targets: Tensor, groups: list[Tensor]) -> Tensor:
        """CE giới hạn trong từng nhóm lớp dễ nhầm: softmax chỉ trên logit của nhóm, nên các lớp trong nhóm phải tách nhau (hard negative) mà không bị lớp ngoài nhóm làm loãng.

        targets: chỉ số lớp (B,) hoặc nhãn mềm (B, C) sau mixup. Mẫu có phần khối lượng nhãn nằm trong nhóm được tính với trọng số bằng phần đó,
        nhãn trong nhóm được chuẩn hoá lại. Không có mẫu nào thuộc nhóm thì loss bằng 0. Chia cho B để cùng thang với CE."""
        soft = targets if targets.dtype.is_floating_point else F.one_hot(targets, logits.shape[1]).float()
        total = logits.new_zeros(())
        for group in groups:
            group = group.to(logits.device)
            mass = soft[:, group].sum(dim=1)
            inside = soft[:, group] / mass.clamp_min(1e-8)[:, None]
            nll = -(inside * F.log_softmax(logits[:, group].float(), dim=1)).sum(dim=1)
            total = total + (mass * nll).sum()
        return total / len(logits)


    return CRDLoss, QKVLoss, hard_negative_loss, pool_teacher, resolve_groups


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hard-negative theo nhóm lớp dễ nhầm (L_hn)

    Ma trận nhầm lẫn của student trên validation (checkpoint tốt nhất, epoch 27, macro recall 0,3946) cho thấy lỗi tập trung ở các lớp có âm sắc hoặc tính chất gần nhau, chứ không chỉ do thiên lệch prior:

    | Nhóm (`HN_GROUPS`, cell 1) | Cặp nhầm điển hình trên val |
    | --- | --- |
    | Saxophone, Clarinet, Trumpet, Flute, Harmonica, Violin | Saxophone → Clarinet 32%, Saxophone → Trumpet 18%, Trumpet → Harmonica 17%, Clarinet → Flute 16% |
    | Glockenspiel, Chime, Electric piano, Acoustic guitar, Telephone | Chime → Telephone 23%, Glockenspiel → Electric piano 17%, Acoustic guitar → Electric piano 17% |
    | Knock, Gunshot, Shatter, Fireworks | Knock, Shatter, Fireworks → Gunshot 16-17% |
    | Scissors, Tearing, Writing, Keys jangling, Computer keyboard | Scissors ↔ Tearing, Writing → Scissors 16% |
    | Microwave oven, Gong, Bus | Microwave oven → Gong 22%, → Bus 14% |
    | Fart, Meow | Fart → Meow 14% |

    Vì các nhóm được chọn từ ma trận nhầm lẫn trên validation, phần cải thiện trên chính các lớp này một phần là do chọn nhóm theo validation; số liệu test mới là phép đo độc lập.

    **Loss.** Với mỗi nhóm g, softmax chỉ trên logit của các lớp trong g, nên các lớp trong nhóm phải tách nhau mà không bị lớp ngoài nhóm làm loãng:

    `L_hn = (1/B) · Σ_g Σ_i m_i^g · CE( softmax(z_i[g]), ỹ_i[g] )`, với `m_i^g` là khối lượng nhãn của mẫu i nằm trong nhóm g và `ỹ_i[g]` là nhãn đã chuẩn hoá lại trong nhóm.

    Nhãn mềm sau mixup được xử lý đúng (phần khối lượng nằm ngoài nhóm không bị tính), mẫu không thuộc nhóm nào cho loss bằng 0 và gradient chỉ rơi vào logit của lớp trong nhóm. Mỗi lớp thuộc nhiều nhất một nhóm; tên lớp được khớp với tên đầy đủ hoặc tiền tố trong dataset và báo lỗi nếu thiếu hoặc mơ hồ (`resolve_groups`). L_hn không dùng teacher và không nhân hệ số KD, nên bật `hn` trong `RUNS` là một thí nghiệm độc lập với ablation ba loss KD.

    Đây là hướng theo **biểu diễn**: hiệu chỉnh bias không khắc phục được các cặp nhầm này (xem mục hiệu chỉnh ở cuối notebook).
    """)
    return


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
    return DATASET_ROOT, time


@app.cell
def _(
    Dataset,
    F,
    NHAT_CACHE_DIR,
    NUM_CLASSES,
    Path,
    RUN_TRAINING,
    Subset,
    Tensor,
    USE_NHAT_CACHE,
    np,
    torch,
):
    # (6) Dữ liệu: train có nhiễu thật và chỉ số mẫu (cho memory bank CRD)
    from scripts.finetune import SMOKE_BATCH, SMOKE_SAMPLES
    from scripts.load_dataset import Mixture36Dataset, build_dataloader, subsample

    DURATION = 4.0


    def nhat_path(split: str) -> Path:
        return NHAT_CACHE_DIR / f"{split}.npy"


    class NoiseCache:
        """n^ đã tính trước, đọc qua memmap; mỗi worker mở riêng."""

        def __init__(self, split: str) -> None:
            self.path, self._array = nhat_path(split), None

        def __getitem__(self, index: int) -> Tensor:
            if self._array is None:
                self._array = np.load(self.path, mmap_mode="r")
            return torch.from_numpy(np.array(self._array[index]))


    class IndexedDataset(Dataset):
        def __init__(self, base: Dataset, cache: NoiseCache | None = None) -> None:
            self.base, self.cache = base, cache

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> dict:
            item = dict(self.base[index])
            item["index"] = torch.tensor(index)
            if self.cache is not None:
                original = self.base.indices[index] if isinstance(self.base, Subset) else index
                item["noise_hat"] = self.cache[original]
            return item


    def with_fork(loader):
        """Kernel marimo mặc định tạo worker bằng spawn, không pickle được IndexedDataset (định nghĩa trong cell). Worker chỉ đọc tệp, không dùng CUDA, nên fork an toàn."""
        if loader.num_workers > 0:
            loader.multiprocessing_context = "fork"
        return loader


    def rms(x: Tensor) -> Tensor:
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)


    def augment_pair(noise_hat: Tensor, noise: Tensor, labels: Tensor, aug: dict) -> tuple[Tensor, Tensor, Tensor, bool]:
        """Biến đổi NHẤT QUÁN cặp (n^ cho student, nhiễu thật cho teacher) để teacher vẫn cho nhãn và q/k/v đúng cho đầu vào của student.

        1. Đưa từng tín hiệu về RMS = 1. Năng lượng n^ lệch so với nhiễu thật tuỳ SNR và tuỳ clip (FRCRN giữ khoảng 95% ở -5 dB nhưng chỉ
           khoảng 65% ở 20 dB), nên nếu trộn thẳng thì cân bằng giữa hai nguồn ở student và ở teacher sẽ khác nhau. Gain đơn thuần thì không
           ảnh hưởng vì log-mel và teacher đều chuẩn hoá theo clip, nên không cần gain ngẫu nhiên riêng.
        2. Dịch vòng cùng một lượng cho cả hai tín hiệu.
        3. Mixup: trộn với cùng hệ số lam (lam >= 0.5, mẫu gốc luôn chiếm ưu thế) và cùng hoán vị; nhãn mềm lam*y_i + (1-lam)*y_j.
        """
        noise_hat, noise = noise_hat / rms(noise_hat), noise / rms(noise)
        batch, samples = noise.shape
        if aug["shift"]:
            start = torch.randint(0, samples, (batch, 1), device=noise.device)
            index = (torch.arange(samples, device=noise.device)[None, :] + start) % samples
            noise_hat, noise = noise_hat.gather(1, index), noise.gather(1, index)
        targets = F.one_hot(labels, NUM_CLASSES).float()
        mixed = bool(aug["mixup_prob"] > 0 and torch.rand(()) < aug["mixup_prob"])
        if mixed:
            lam = torch.distributions.Beta(aug["mixup_alpha"], aug["mixup_alpha"]).sample((batch, 1)).to(noise.device)
            lam = torch.maximum(lam, 1.0 - lam)
            order = torch.randperm(batch, device=noise.device)
            noise_hat = lam * noise_hat + (1.0 - lam) * noise_hat[order]
            noise = lam * noise + (1.0 - lam) * noise[order]
            targets = lam * targets + (1.0 - lam) * targets[order]
        return noise_hat, noise, targets, mixed


    def use_cache() -> bool:
        return USE_NHAT_CACHE and RUN_TRAINING  # smoke test không dùng cache


    def cache_ready(split: str) -> bool:
        return use_cache() and (NHAT_CACHE_DIR / f"{split}.done").is_file()  # test không cache: chỉ dùng một lần, FRCRN chạy trực tiếp


    def build_eval_loader(split: str, batch_size: int, workers: int, max_samples: int | None = None):
        base = subsample(Mixture36Dataset(split=split, duration=DURATION), max_samples)
        dataset = IndexedDataset(base, NoiseCache(split)) if cache_ready(split) else base
        return with_fork(build_dataloader(dataset, batch_size, False, workers))


    def build_loaders(train_config: dict, max_samples: int | None, workers: int) -> dict:
        base = subsample(Mixture36Dataset(split="train", duration=DURATION, load_noise=True), max_samples)
        train_ds = IndexedDataset(base, NoiseCache("train") if use_cache() else None)
        return {
            "train": with_fork(build_dataloader(train_ds, train_config["batch_size"], True, workers)),
            "validation": build_eval_loader("validation", train_config["eval_batch_size"], workers, max_samples),
            "num_train": len(train_ds),
        }

    return (
        DURATION,
        Mixture36Dataset,
        NoiseCache,
        SMOKE_BATCH,
        SMOKE_SAMPLES,
        augment_pair,
        build_dataloader,
        build_eval_loader,
        build_loaders,
        nhat_path,
        use_cache,
        with_fork,
    )


@app.cell
def _(
    DURATION,
    FRONTEND,
    Mixture36Dataset,
    NHAT_CACHE_DIR,
    NHAT_DRIVE_ZIP,
    NHAT_ZIP,
    NUM_WORKERS,
    NoiseCache,
    ROOT,
    Subset,
    UPLOAD_NHAT_TO_DRIVE,
    build_dataloader,
    device,
    fs,
    nhat_path,
    np,
    student,
    time,
    torch,
    tqdm,
    use_cache,
    with_fork,
):
    # (6b) Tính trước n^ = FRCRN(mixture) cho train và validation (test chỉ dùng một lần nên chạy FRCRN trực tiếp). Chạy một lần (khoảng 17 phút), lần sau bỏ qua nhờ tệp .done.
    import zipfile

    from numpy.lib.format import open_memmap


    def build_nhat_cache(splits=("train", "validation")) -> None:
        NHAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        separator = student.separator.eval()
        for split in splits:
            done, lock = NHAT_CACHE_DIR / f"{split}.done", NHAT_CACHE_DIR / f"{split}.building"
            dataset = Mixture36Dataset(split=split, duration=DURATION)
            if done.is_file() and done.read_text() == str(len(dataset)):
                print(f"n^ cache {split}: đã có ({len(dataset)} clip), bỏ qua")
                continue
            if lock.is_file() and time.time() - lock.stat().st_mtime < 600:  # hai tiến trình cùng ghi một tệp đã làm hỏng cache một lần
                raise RuntimeError(f"đang có tiến trình khác tính n^ {split} ({lock}), đợi nó xong rồi chạy lại")
            lock.write_text("building")
            samples = int(DURATION * FRONTEND["sample_rate"])
            out = open_memmap(nhat_path(split), mode="w+", dtype=np.float32, shape=(len(dataset), samples))
            position, started = 0, time.perf_counter()
            for step, batch in enumerate(tqdm(with_fork(build_dataloader(dataset, 64, False, NUM_WORKERS)), desc=f"n^ {split}", unit="batch")):
                with torch.no_grad():
                    estimate = separator(batch["waveform"].to(device).float()).cpu().numpy()
                out[position:position + len(estimate)] = estimate
                position += len(estimate)
                if step % 20 == 0:
                    lock.touch()
            out.flush()
            done.write_text(str(len(dataset)))
            lock.unlink(missing_ok=True)
            print(f"n^ cache {split}: {len(dataset)} clip trong {time.perf_counter() - started:.0f}s -> {nhat_path(split)}")


    def empty_rows(split: str) -> np.ndarray:
        """Hàng toàn số 0 trong cache (chưa được ghi)."""
        array, rows = np.load(nhat_path(split), mmap_mode="r"), []
        for start in range(0, len(array), 512):
            energy = np.square(np.asarray(array[start:start + 512], dtype=np.float64)).sum(axis=1)
            rows.extend((start + np.nonzero(energy == 0)[0]).tolist())
        return np.array(rows, dtype=np.int64)


    def fix_empty_rows(split: str) -> None:
        rows = empty_rows(split)
        if not len(rows):
            print(f"n^ cache {split}: không có hàng trống")
            return
        print(f"n^ cache {split}: {len(rows)} hàng trống ({len(rows) / len(np.load(nhat_path(split), mmap_mode='r')):.1%}), tính lại riêng các hàng này")
        array = np.load(nhat_path(split), mmap_mode="r+")
        subset = Subset(Mixture36Dataset(split=split, duration=DURATION), rows.tolist())
        position = 0
        for batch in tqdm(with_fork(build_dataloader(subset, 64, False, NUM_WORKERS)), desc=f"n^ sửa {split}", unit="batch"):
            with torch.no_grad():
                estimate = student.separator.eval()(batch["waveform"].to(device).float()).cpu().numpy()
            array[rows[position:position + len(estimate)]] = estimate
            position += len(estimate)
        array.flush()
        assert not len(empty_rows(split)), f"n^ cache {split}: vẫn còn hàng trống sau khi sửa"


    def check_nhat_cache(count: int = 8) -> None:
        probe = Mixture36Dataset(split="train", duration=DURATION)
        waveform = torch.stack([probe[i]["waveform"] for i in range(count)]).to(device)
        with torch.no_grad():
            live = student.separator(waveform.float()).cpu()
        cached = torch.stack([NoiseCache("train")[i] for i in range(count)])
        error = float((live - cached).abs().max())
        snr = float(10 * torch.log10(live.pow(2).sum() / (live - cached).pow(2).sum().clamp_min(1e-20)))
        print(f"n^ cache: so với FRCRN chạy trực tiếp, sai lệch lớn nhất {error:.2e} (biên độ tối đa {float(live.abs().max()):.3f}), SNR giữa hai bản {snr:.1f} dB")
        # cuDNN TF32 không tất định và phụ thuộc cỡ batch (cùng batch: khoảng 90 dB, khác batch: khoảng 60 dB), nên so bằng SNR chứ không bằng sai số tuyệt đối
        assert snr > 40, "cache n^ không khớp FRCRN trực tiếp"


    def nhat_ready() -> bool:
        return all((NHAT_CACHE_DIR / f"{name}.done").is_file() for name in ("train", "validation"))


    def restore_nhat_from_drive() -> None:
        """Nếu thiếu cache trên máy: tải zip từ Drive rồi giải nén, thay cho việc tính lại bằng FRCRN."""
        if nhat_ready() or fs is None:
            return
        try:
            if not fs.exists(NHAT_DRIVE_ZIP):
                print("n^ cache: Drive chưa có bản lưu, sẽ tính bằng FRCRN")
                return
            print(f"n^ cache: tải {NHAT_DRIVE_ZIP} từ Drive (khoảng 8,6 GB)")
            started = time.perf_counter()
            fs.get_file(NHAT_DRIVE_ZIP, str(NHAT_ZIP))
            with zipfile.ZipFile(NHAT_ZIP) as archive:
                archive.extractall(ROOT)  # các mục có dạng cache/nhat/*.npy
            print(f"n^ cache: đã tải và giải nén trong {time.perf_counter() - started:.0f}s")
        except Exception as error:
            print("drive: không khôi phục được cache n^, sẽ tính bằng FRCRN:", repr(error))


    def save_nhat_to_drive() -> None:
        """Đẩy zip cache lên Drive một lần (bỏ qua nếu Drive đã có). Thư mục tạo một lần, tránh makedirs lặp lại sinh thư mục trùng tên."""
        if fs is None or not UPLOAD_NHAT_TO_DRIVE or not nhat_ready():
            return
        try:
            if fs.exists(NHAT_DRIVE_ZIP):
                print("n^ cache: Drive đã có bản lưu, bỏ qua upload")
                return
            if not NHAT_ZIP.is_file():
                print("n^ cache: nén thành", NHAT_ZIP)
                partial = NHAT_ZIP.with_name(NHAT_ZIP.name + ".part")
                with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                    for path in sorted(NHAT_CACHE_DIR.glob("*")):
                        archive.write(path, path.relative_to(ROOT).as_posix())
                partial.replace(NHAT_ZIP)
            fs.makedirs(NHAT_DRIVE_ZIP.rsplit("/", 1)[0], exist_ok=True)
            print(f"n^ cache: upload {NHAT_ZIP.stat().st_size / 1e9:.1f} GB lên Drive ({NHAT_DRIVE_ZIP})")
            started = time.perf_counter()
            fs.put_file(str(NHAT_ZIP), NHAT_DRIVE_ZIP)
            print(f"n^ cache: upload xong sau {time.perf_counter() - started:.0f}s")
        except Exception as error:
            print("drive: upload cache n^ lỗi (huấn luyện vẫn tiếp tục):", repr(error))


    if use_cache():
        restore_nhat_from_drive()
        build_nhat_cache()
        for split_name in ("train", "validation"):
            fix_empty_rows(split_name)  # chạy lại lần sau vẫn an toàn: không có hàng trống thì bỏ qua (quét khoảng 10 giây)
        check_nhat_cache()
        save_nhat_to_drive()
    else:
        print("n^ cache: tắt (USE_NHAT_CACHE=False hoặc smoke test), FRCRN chạy trực tiếp mỗi bước")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kiểm tra teacher (tuỳ chọn, rẻ)

    Teacher phải phân loại nhiễu thật đúng cỡ `clean_val_accuracy` trong checkpoint (fold 0: 0.74). Nếu thấp hơn nhiều thì đường vào (RMS, fbank) lệch so với lúc fine-tune, nên dừng lại trước khi huấn luyện student.
    """)
    return


@app.cell
def _(DURATION, Mixture36Dataset, build_dataloader, device, logger, teacher):
    # (7) Độ chính xác của teacher trên nhiễu thật của tập validation
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Huấn luyện và đánh giá

    **Vì sao có hàng đợi thí nghiệm.** Thí nghiệm tham chiếu (cả ba loss KD, 50 epoch, tên không có hậu tố) dừng quanh epoch 33: macro_acc trên validation đi ngang ở khoảng 0,39 từ epoch ~22 (tốt nhất 0,3946 ở epoch 27), train acc 0,75 so với val 0,39. Teacher fold 0 chạy trên chính n̂ đạt 0,6633, nên đầu vào không phải trần và cần xem thành phần KD nào thực sự giúp. Cell (9) chạy lần lượt các nhánh trong `RUNS` (cell 1) rồi in bảng so sánh.

    | Nhánh | Khác tham chiếu | Câu hỏi |
    | --- | --- | --- |
    | `eskd` (tuỳ chọn, mặc định tắt) | khởi tạo từ `best_state` tham chiếu, CRD tắt, KD (q/k/v và logit) giảm tuyến tính 1 → 0 trong 8 epoch, lr 1e-5, 10 epoch; epoch 0 đo điểm xuất phát nên best không thể tệ hơn; từ khi KD hết thì không gọi teacher | bỏ dần teacher có cải thiện val không |
    | `no-crd` | tắt L_crd | CRD có giúp không |
    | `no-qkv` | tắt L_qkv | q/k/v có giúp không |
    | `ce` | tắt cả ba loss KD | KD có giúp so với CE thuần không |
    | `hn` | bật L_hn (λ = 1) | hard-negative có tách được các cặp nhầm không |

    `eskd` là nhánh duy nhất không huấn luyện từ đầu (nó khởi tạo từ `best_state` của tham chiếu), nên được chú thích trong `RUNS`; bỏ chú thích để dùng. Có thể thêm nhánh `T2` (nhiệt độ 2) và `qkv9-12` (chỉ q/k/v của 4 lớp cuối) bằng cách bỏ chú thích trong cell 1.

    **Giữ phép so sánh công bằng.**

    - Nhánh ablation dừng ở `max_epochs = 15` nhưng **không** đổi `TRAIN["epochs"]`, nên lịch warm-up và lr giống tham chiếu; cùng seed nên so cặp được. Bảng so sánh lấy đường cong của tham chiếu từ `history` trong `resume.pt` của nó và so ở cùng ngân sách epoch (`ref_same_budget`) và với tốt nhất toàn bộ (`ref_overall`).
    - Val có 6.480 mẫu; sai số chuẩn của macro_acc khoảng 0,006 và dao động giữa các epoch khoảng ±0,005, nên chênh dưới ~0,02 giữa hai nhánh một seed chưa đáng tin.
    - Các nhánh ablation đặt `eval_test = False`: **không động tới test**. Test chỉ chạy một lần cho cấu hình được chọn sau cùng (đặt `eval_test = True`).
    - Tên nhánh = tên cơ sở + `-<tag>-<RUN_TAG>` nên không đụng `resume.pt` và thư mục Drive của tham chiếu; mỗi nhánh có `resume.pt` riêng nên Run all lại sẽ nối tiếp. Các khoá mới chỉ vào cờ so khớp khi khác mặc định, nên `resume.pt` của tham chiếu vẫn nối tiếp được.

    **Thời gian.** Khoảng 3 phút mỗi epoch có teacher (gồm upload Drive); hàng đợi mặc định 4 × 15 = 60 epoch, khoảng 3 giờ trên GPU sandbox. Bớt dòng trong `RUNS` để rút ngắn.
    """)
    return


@app.cell
def _(
    AUG,
    AdamW,
    CRDLoss,
    DRIVE_DIR,
    DURATION,
    EMBED_CHANNELS,
    F,
    FRCRNMobileViT,
    FRCRN_PATH,
    GRID,
    HN_GROUPS,
    KD,
    Mixture36Dataset,
    NUM_CLASSES,
    NUM_WORKERS,
    Path,
    QKVLoss,
    ROOT,
    RUN_TAG,
    RUN_TRAINING,
    SMOKE_BATCH,
    SMOKE_SAMPLES,
    SNR_LEVELS,
    STUDENT,
    TEACHER_LAYERS,
    TRAIN,
    Tensor,
    USE_AUG,
    USE_CRD,
    USE_LOGIT,
    USE_QKV,
    augment_pair,
    build_eval_loader,
    build_loaders,
    build_student,
    device,
    fs,
    hard_negative_loss,
    logger,
    nn,
    np,
    pd,
    pool_teacher,
    resolve_groups,
    run,
    set_seed,
    teacher,
    time,
    torch,
    tqdm,
    use_cache,
):
    # (8) Vòng huấn luyện + đánh giá 9 chỉ số theo từng SNR
    from src.metrics import ClassificationMetrics  # noqa: F401  (snr_scores dùng bên dưới)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from src.train.reviewkd_trainer import NINE_METRICS, snr_scores


    class DrivePut:
        """Upload một tệp lên Drive, ghi đè nếu đã có. Thư mục chỉ tạo MỘT lần (ensure): makedirs lặp lại ở mỗi lần upload đã sinh thư mục trùng tên
        trên Drive và gây MultipleFilesError. Lỗi mạng chỉ ghi cảnh báo, không làm gãy huấn luyện."""

        ready: set[str] = set()

        def __init__(self, fs, remote_dir: str, filename: str, logger, attempts: int = 3) -> None:
            self.fs, self.remote_dir, self.filename, self.logger, self.attempts = fs, remote_dir.strip("/"), filename, logger, attempts

        @classmethod
        def ensure(cls, fs, remote_dir: str, logger) -> bool:
            remote_dir = remote_dir.strip("/")
            if remote_dir in cls.ready:
                return True
            try:
                fs.makedirs(remote_dir, exist_ok=True)
                cls.ready.add(remote_dir)
                return True
            except Exception as error:
                logger.warning("drive: không tạo được thư mục %s: %r", remote_dir, error)
                return False

        def __call__(self, local_path, epoch: int | None = None, score: float | None = None) -> None:
            if self.remote_dir not in self.ready:
                return  # thư mục chưa sẵn sàng thì bỏ qua, không thử tạo lại
            target = f"{self.remote_dir}/{self.filename}"
            for attempt in range(1, self.attempts + 1):
                try:
                    self.fs.put_file(str(local_path), target)
                    self.logger.info("drive backup: %s -> %s", Path(local_path).name, target)
                    return
                except Exception as error:
                    if attempt == self.attempts:
                        self.logger.warning("drive backup failed for %s: %r", local_path, error)
                    else:
                        time.sleep(5 * attempt)
    from src.metrics import _binary_ap, _binary_auc, confusion_matrix, top_confusions  # hàm nội bộ của repo


    def trainable_state(model: nn.Module) -> dict[str, Tensor]:
        """Trọng số cần lưu: backbone (net) và attention pooling (att, cla). Bỏ separator (FRCRN đóng băng, nạp từ pretrained).
        Trước đây chỉ lưu model.net nên thiếu att/cla: best.pt không phân loại được và resume làm mất lớp phân loại."""
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if not key.startswith("separator.")}


    def load_trainable_state(model: nn.Module, state: dict[str, Tensor]) -> None:
        result = model.load_state_dict(state, strict=False)
        problems = [key for key in result.missing_keys if not key.startswith("separator.")] + list(result.unexpected_keys)
        if problems:
            raise ValueError(f"trọng số không khớp mô hình (thiếu hoặc thừa): {problems[:6]}")


    def save_model(path: Path, model: nn.Module, **metadata) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": trainable_state(model), **metadata}, path)
        return path


    @torch.no_grad()
    def evaluate(model: FRCRNMobileViT, loader, desc: str = "evaluate") -> dict[str, float]:
        model.eval()
        probabilities, labels, snrs = [], [], []
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            if "noise_hat" in batch:
                probabilities.append(model.predict_proba(noise_hat=batch["noise_hat"].to(device)).float().cpu())
            else:
                probabilities.append(model.predict_proba(batch["waveform"].to(device)).float().cpu())
            labels.append(batch["target"].argmax(dim=1))
            snrs.append(batch["snr_db"])
        return snr_scores(torch.cat(probabilities).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), NUM_CLASSES)


    def per_snr_table(scores: dict[str, float]) -> pd.DataFrame:
        rows = {f"{level}": [scores.get(f"{name}_snr_{level}", float("nan")) for name in NINE_METRICS] for level in SNR_LEVELS}
        rows["mean"] = [scores.get(f"{name}_snr_mean", float("nan")) for name in NINE_METRICS]
        return pd.DataFrame.from_dict(rows, orient="index", columns=NINE_METRICS).rename_axis("snr_db")


    @torch.no_grad()
    def analyze_predictions(model: nn.Module, loader, split: str, name: str, out_dir: Path, class_names: list[str]) -> list[Path]:
        """Dự đoán của mô hình tốt nhất: xác suất từng clip, ma trận nhầm lẫn, bảng theo từng lớp (recall, precision, AP, AUC, lớp hay bị nhầm thành, recall theo SNR)."""
        model.eval()
        probs, labels, snrs = [], [], []
        for batch in tqdm(loader, desc=f"{name} - dự đoán {split}", unit="batch", leave=False, dynamic_ncols=True):
            if "noise_hat" in batch:
                probs.append(model.predict_proba(noise_hat=batch["noise_hat"].to(device)).float().cpu())
            else:
                probs.append(model.predict_proba(batch["waveform"].to(device)).float().cpu())
            labels.append(batch["target"].argmax(dim=1))
            snrs.append(batch["snr_db"])
        P, y, snr = torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy()
        pred = P.argmax(axis=1)
        matrix = confusion_matrix(y, pred, NUM_CLASSES)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [out_dir / f"predictions_{split}_{name}.npz", out_dir / f"confusion_{split}_{name}.csv", out_dir / f"per_class_{split}_{name}.csv"]
        np.savez_compressed(paths[0], probs=P.astype(np.float16), labels=y, snr_db=snr, class_names=np.array(class_names))
        pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(paths[1])
        rows = []
        for c in range(NUM_CLASSES):
            off_diagonal = matrix[c].copy()
            off_diagonal[c] = 0
            partner, total = int(off_diagonal.argmax()), max(int(matrix[c].sum()), 1)
            row = dict(label=class_names[c], clips=int(matrix[c].sum()), recall=matrix[c, c] / total, precision=matrix[c, c] / max(int(matrix[:, c].sum()), 1),
                       AP=_binary_ap((y == c).astype(np.int64), P[:, c]), AUC=_binary_auc((y == c).astype(np.int64), P[:, c]),
                       most_confused_as=class_names[partner], confused_rate=off_diagonal[partner] / total)
            for level in SNR_LEVELS:
                selected = (y == c) & (snr == level)
                row[f"recall_snr_{level}"] = float((pred[selected] == c).mean()) if selected.any() else float("nan")
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("recall")
        table.to_csv(paths[2], index=False)
        logger.info("%s %s: 12 cặp nhầm nhiều nhất (thật -> đoán, tỉ lệ trên lớp thật)\n%s", name, split,
                    "\n".join(f"  {p.true_label} -> {p.predicted_label}: {p.rate:.0%} ({p.count} clip)" for p in top_confusions(matrix, class_names, 12)))
        logger.info("%s %s: 8 lớp yếu nhất theo recall\n%s", name, split, table.head(8)[["label", "recall", "precision", "AP", "most_confused_as", "confused_rate"]].round(3).to_string(index=False))
        return paths


    def kd_scale(epoch: int, warmup: int) -> float:
        return 1.0 if warmup == 0 else min(1.0, (epoch - 1) / warmup)


    DEFAULT_CFG = dict(tag="", qkv=None, crd=None, logit=None, T=None, qkv_layers=None, hn=0.0, init_from=None,
                       kd_decay_epochs=None, lr=None, warmup_ratio=None, max_epochs=None, eval_test=True)


    def resolve_cfg(cfg: dict | None) -> dict:
        """Gộp cấu hình một lần chạy với mặc định. qkv, crd, logit lấy từ cờ ở cell (1) nếu để None."""
        cfg = {**DEFAULT_CFG, **(cfg or {})}
        unknown = set(cfg) - set(DEFAULT_CFG)
        if unknown:
            raise KeyError(f"khoá không hợp lệ trong RUNS: {sorted(unknown)}, hợp lệ: {sorted(DEFAULT_CFG)}")
        for key, default in (("qkv", USE_QKV), ("crd", USE_CRD), ("logit", USE_LOGIT)):
            cfg[key] = default if cfg[key] is None else cfg[key]
        return cfg


    def run_name(cfg: dict | None = None) -> str:
        """Tên thí nghiệm. Cấu hình tham chiếu (cfg rỗng) giữ nguyên tên cũ, mọi cấu hình khác có thêm `-<tag>-<RUN_TAG>` để không đụng resume.pt và thư mục Drive cũ."""
        cfg = resolve_cfg(cfg)
        base = "-".join(["mobilevit12", f"ffn{STUDENT['ffn_dim']}", f"lr{TRAIN['lr']:.0e}-do{round(TRAIN['dropout'] * 100)}-dp{round(TRAIN['drop_path'] * 100)}-wd{round(TRAIN['weight_decay'] * 100)}", "frcrn",
                         *([f"mx{round(AUG['mixup_prob'] * 100)}"] if USE_AUG else []), *(["qkv"] if cfg["qkv"] else []), *(["crd"] if cfg["crd"] else []), *(["logit"] if cfg["logit"] else []),
                         *([] if (cfg["qkv"] or cfg["crd"] or cfg["logit"]) else ["ce"])])
        return base if cfg == resolve_cfg(None) else f"{base}-{cfg['tag'] or 'x'}-{RUN_TAG}"


    def compare_runs(results: list[dict]) -> pd.DataFrame:
        """So từng thí nghiệm với thí nghiệm tham chiếu (cả ba loss KD) ở CÙNG ngân sách epoch; cùng seed nên so cặp được. Tham chiếu lấy từ history trong resume.pt."""
        path = ROOT / "output" / "resume" / run_name(None) / "resume.pt"
        curve = [r["val_macro_acc"] for r in torch.load(path, map_location="cpu", weights_only=False)["history"] if r["epoch"] > 0] if path.is_file() else []
        rows = []
        for r in results:
            budget = len(r["val_curve"])
            same = max(curve[:budget]) if curve else float("nan")
            overall = max(curve) if curve else float("nan")
            rows.append({"run": r["model"], "epochs": budget, "best_epoch": r["best_epoch"], "best_val": r["val_macro_acc"],
                         "ref_same_budget": same, "delta_same_budget": r["val_macro_acc"] - same,
                         "ref_overall": overall, "delta_overall": r["val_macro_acc"] - overall})
        return pd.DataFrame(rows).round(4)


    def run_experiment(name: str, cfg: dict | None = None) -> tuple[dict, pd.DataFrame | None]:
        cfg = resolve_cfg(cfg)
        use_qkv, use_crd, use_logit = cfg["qkv"], cfg["crd"], cfg["logit"]
        train_config, kd = dict(TRAIN), dict(KD)
        if cfg["T"] is not None:
            kd["logit_temperature"] = cfg["T"]
        if cfg["lr"] is not None:
            train_config["lr"] = cfg["lr"]
        if cfg["warmup_ratio"] is not None:
            train_config["warmup_ratio"] = cfg["warmup_ratio"]
        eval_test = RUN_TRAINING and cfg["eval_test"]  # test chỉ chạy cho cấu hình đã chọn, các nhánh ablation đặt eval_test=False
        class_names = Mixture36Dataset(split="validation", duration=DURATION).class_names
        hn_groups = resolve_groups(HN_GROUPS, class_names) if cfg["hn"] else []
        qkv_layers = [i for i in range(len(TEACHER_LAYERS)) if cfg["qkv_layers"] is None or (i + 1) in cfg["qkv_layers"]]  # chỉ số lớp (từ 0) có loss q/k/v
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
                       kd["crd_temperature"], kd["crd_momentum"]).to(device) if use_crd else None)

        trainable = [p for module in (model, crd) if module is not None for p in module.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=train_config["lr"], weight_decay=train_config["weight_decay"], betas=train_config["betas"])
        steps_per_epoch = len(loaders["train"])
        warmup_steps = max(int(train_config["epochs"] * steps_per_epoch * train_config["warmup_ratio"]), 1)  # warm-up tuyến tính theo bước
        plateau = ReduceLROnPlateau(optimizer, mode="max", factor=train_config["lr_factor"], patience=train_config["lr_patience"],
                                    threshold=train_config["lr_threshold"], threshold_mode="abs", min_lr=train_config["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = train_config["lr"] / warmup_steps  # bước đầu tiên của warm-up; nạp resume sẽ ghi đè bằng lr đã lưu
        logger.info("%s | student %.2fM | crd %.2fM | train %d mẫu", name,
                    sum(p.numel() for p in model.net.parameters()) / 1e6,
                    sum(p.numel() for p in crd.parameters()) / 1e6 if crd else 0.0, loaders["num_train"])

        history, best_score, best_epoch, best_state, stale, start_epoch = [], -float("inf"), 0, None, 0, 1
        flags = dict(qkv=use_qkv, crd=use_crd, logit=use_logit, layers=TEACHER_LAYERS, student=STUDENT, kd=kd,
                     reg=dict(dropout=TRAIN["dropout"], drop_path=TRAIN["drop_path"], wd=TRAIN["weight_decay"]),
                     aug=(AUG if USE_AUG else None),
                     opt=dict(lr=train_config["lr"], factor=TRAIN["lr_factor"], patience=TRAIN["lr_patience"], threshold=TRAIN["lr_threshold"],
                              min_lr=TRAIN["min_lr"], warmup=train_config["warmup_ratio"]), cache=use_cache(), fmt=2)
        # Núm mới chỉ vào flags khi khác mặc định, để resume.pt của cấu hình tham chiếu (không có các khoá này) vẫn khớp.
        extras = dict(T=cfg["T"], qkv_layers=cfg["qkv_layers"], hn=cfg["hn"] or None, init=cfg["init_from"], decay=cfg["kd_decay_epochs"])
        flags.update({key: value for key, value in extras.items() if value})
        resume_path = ROOT / "output" / "resume" / name / "resume.pt" if RUN_TRAINING else None  # thư mục cố định để Run all nối tiếp được
        remote_dir = f"{DRIVE_DIR}/{name}"
        use_drive = fs is not None and RUN_TRAINING
        drive_ready = use_drive and DrivePut.ensure(fs, remote_dir, logger)  # tạo thư mục Drive một lần duy nhất
        drive_best = DrivePut(fs, remote_dir, "best.pt", logger) if drive_ready else None
        drive_resume = DrivePut(fs, remote_dir, "resume.pt", logger) if drive_ready else None
        drive_history = DrivePut(fs, remote_dir, f"history_{name}.csv", logger) if drive_ready else None
        drive_logfile = DrivePut(fs, remote_dir, run.log_file.name, logger) if drive_ready else None
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
            load_trainable_state(model, state["model"])
            if crd is not None:
                crd.load_state_dict(state["crd"])  # gồm cả memory bank
            optimizer.load_state_dict(state["optimizer"])
            plateau.load_state_dict(state["scheduler"])
            history, best_score, best_epoch, best_state, stale = state["history"], state["best_score"], state["best_epoch"], state["best_state"], state["stale"]
            start_epoch = state["epoch"] + 1
            set_seed(train_config["seed"] + start_epoch)
            logger.info("%s: nối tiếp từ epoch %d (best %.4f ở epoch %d, stale %d)", name, start_epoch, best_score, best_epoch, stale)
        elif cfg["init_from"]:  # ESKD: khởi tạo từ best_state của thí nghiệm tham chiếu, đo epoch 0 để best không thể tệ hơn điểm xuất phát
            source = ROOT / "output" / "resume" / run_name(None) / "resume.pt" if cfg["init_from"] == "reference" else Path(cfg["init_from"])
            init = torch.load(source, map_location="cpu", weights_only=False)
            load_trainable_state(model, init["best_state"])
            scores = evaluate(model, loaders["validation"], desc="epoch 0 - validation")
            best_score, best_epoch, best_state = scores["macro_acc"], 0, trainable_state(model)
            history.append({"epoch": 0, **{f"val_{k}": v for k, v in scores.items()}})
            logger.info("%s: khởi tạo từ best_state epoch %d của %s, val macro_acc %.4f", name, init["best_epoch"], source.parent.name, best_score)
        for epoch in range(start_epoch, train_config["epochs"] + 1):
            model.train()
            scale = max(0.0, 1.0 - (epoch - 1) / cfg["kd_decay_epochs"]) if cfg["kd_decay_epochs"] else kd_scale(epoch, kd["warmup_epochs"])
            sums = dict(loss=0.0, ce=0.0, cos=0.0, rel=0.0, crd=0.0, kl=0.0, hn=0.0, correct=0, seen=0)
            cos_layer_sum = None  # 1 - cos(q,k,v) của từng lớp, để thấy lớp nào học chậm
            for batch_index, batch in enumerate(tqdm(loaders["train"], desc=f"epoch {epoch}", unit="batch", leave=False, dynamic_ncols=True)):
                step_index = (epoch - 1) * steps_per_epoch + batch_index
                if step_index < warmup_steps:
                    for group in optimizer.param_groups:
                        group["lr"] = train_config["lr"] * (step_index + 1) / warmup_steps
                noise = batch["noise"].to(device)
                labels, index = batch["target"].argmax(dim=1).to(device), batch["index"].to(device)
                noise_hat, targets, mixed = (batch["noise_hat"].to(device) if "noise_hat" in batch else None), labels, False
                if USE_AUG and noise_hat is not None:
                    noise_hat, noise, targets, mixed = augment_pair(noise_hat, noise, labels, AUG)
                output = model(noise_hat=noise_hat) if noise_hat is not None else model(batch["waveform"].to(device))
                ce = F.cross_entropy(output.logits, targets, label_smoothing=train_config["label_smoothing"])
                loss, cos, rel, nce, kl, hn = ce, ce.new_zeros(()), ce.new_zeros(()), ce.new_zeros(()), ce.new_zeros(()), ce.new_zeros(())
                use_kd = (use_qkv or use_crd or use_logit) and not (cfg["kd_decay_epochs"] and scale <= 0)  # ESKD: hết giảm KD thì bỏ cả lượt teacher
                if use_kd:
                    teacher_output = teacher(noise)
                if use_qkv and use_kd:
                    pairs = [qkv_loss(output.qkv[i], pool_teacher(teacher_output.qkv[i], GRID, output.grids[i])) for i in qkv_layers]
                    cos_layers, rel_layers = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
                    cos, rel = cos_layers.mean(), rel_layers.mean()
                    step_layers = cos_layers.detach() * len(labels)
                    cos_layer_sum = step_layers if cos_layer_sum is None else cos_layer_sum + step_layers
                    loss = loss + scale * kd["qkv_weight"] * (cos + kd["rel_weight"] * rel)
                if use_crd and use_kd and not mixed:  # batch trộn không có chỉ số mẫu và nhãn đơn nên không cập nhật memory bank của CRD
                    nce = crd(output.embedding, teacher_output.embedding, index, labels)
                    loss = loss + scale * kd["crd_weight"] * nce
                if use_logit and use_kd:
                    temperature = kd["logit_temperature"]
                    kl = F.kl_div(F.log_softmax(output.logits / temperature, dim=-1),
                                  F.softmax(teacher_output.logits / temperature, dim=-1), reduction="batchmean") * temperature**2
                    loss = loss + scale * kd["logit_weight"] * kl
                if hn_groups:  # hard-negative: CE giới hạn trong nhóm lớp dễ nhầm, không dùng teacher và không nhân kd scale nên độc lập với ablation KD và với ESKD
                    hn = hard_negative_loss(output.logits, targets, hn_groups)
                    loss = loss + cfg["hn"] * hn
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, train_config["grad_clip"])
                optimizer.step()
                n = len(labels)
                for key, value in (("loss", loss), ("ce", ce), ("cos", cos), ("rel", rel), ("crd", nce), ("kl", kl), ("hn", hn)):
                    sums[key] += value.item() * n
                sums["correct"] += (output.logits.argmax(1) == labels).sum().item()
                sums["seen"] += n
            seen = max(sums["seen"], 1)
            scores = evaluate(model, loaders["validation"], desc=f"epoch {epoch} - validation")
            record = {"epoch": epoch, "kd_scale": scale, "lr": optimizer.param_groups[0]["lr"], "train_acc": sums["correct"] / seen,
                      **{f"train_{k}": sums[k] / seen for k in ("loss", "ce", "cos", "rel", "crd", "kl", "hn")},
                      **({f"train_cos_l{i + 1}": v / seen for i, v in zip(qkv_layers, cos_layer_sum.tolist())} if cos_layer_sum is not None else {}),
                      **{f"val_{k}": v for k, v in scores.items()}}
            history.append(record)
            logger.info("%s epoch %d | loss %.4f ce %.4f cos %.4f rel %.4f crd %.4f kl %.4f hn %.4f | train acc %.4f | val macro_acc %.4f mAP_macro %.4f | lr %.2e",
                        name, epoch, record["train_loss"], record["train_ce"], record["train_cos"], record["train_rel"],
                        record["train_crd"], record["train_kl"], record["train_hn"], record["train_acc"], scores["macro_acc"], scores["mAP_macro"], record["lr"])
            if (epoch * steps_per_epoch) >= warmup_steps:  # chỉ giảm lr sau warm-up
                plateau.step(scores["macro_acc"])
                if optimizer.param_groups[0]["lr"] < record["lr"]:
                    logger.info("%s: val macro_acc không cải thiện, giảm lr %.2e -> %.2e", name, record["lr"], optimizer.param_groups[0]["lr"])
            if RUN_TRAINING:
                run.result_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(run.result_dir / f"history_{name}.csv", index=False)
                if drive_history is not None:
                    drive_history(run.result_dir / f"history_{name}.csv")
            if scores["macro_acc"] > best_score:
                best_score, best_epoch, stale = scores["macro_acc"], epoch, 0
                best_state = trainable_state(model)
                if RUN_TRAINING:  # lưu ngay khi tốt hơn, để sandbox mất kết nối giữa chừng vẫn còn checkpoint
                    save_model(run.checkpoint_path(name), model, epoch=epoch, macro_acc=best_score, frcrn=str(FRCRN_PATH.relative_to(ROOT)))
                    if drive_best is not None:
                        drive_best(run.checkpoint_path(name), epoch, best_score)  # ghi đè best.pt trên Drive
            else:
                stale += 1
            if resume_path is not None:  # ghi nguyên tử, để bị ngắt giữa lúc ghi vẫn còn bản cũ
                resume_path.parent.mkdir(parents=True, exist_ok=True)
                partial = resume_path.with_name(resume_path.name + ".part")
                torch.save({"epoch": epoch, "flags": flags, "model": trainable_state(model),
                            "crd": crd.state_dict() if crd is not None else None,
                            "optimizer": optimizer.state_dict(), "scheduler": plateau.state_dict(), "history": history,
                            "best_score": best_score, "best_epoch": best_epoch, "best_state": best_state, "stale": stale}, partial)
                partial.replace(resume_path)
                if drive_resume is not None:
                    drive_resume(resume_path)
                if drive_logfile is not None:
                    drive_logfile(run.log_file)
            if stale >= train_config["patience"]:
                logger.info("%s: dừng sớm ở epoch %d (best %d, %.4f)", name, epoch, best_epoch, best_score)
                break
            if cfg["max_epochs"] and epoch >= cfg["max_epochs"]:  # ngân sách epoch của nhánh ablation; lịch lr và warm-up vẫn theo TRAIN["epochs"] nên so được với tham chiếu
                logger.info("%s: dừng ở epoch %d theo max_epochs (best %d, %.4f)", name, epoch, best_epoch, best_score)
                break

        load_trainable_state(model, best_state)
        best_record = next(r for r in history if r["epoch"] == best_epoch)
        val_table = per_snr_table({k.removeprefix("val_"): v for k, v in best_record.items() if k.startswith("val_")})
        logger.info("%s: validation tại epoch tốt nhất %d, 9 chỉ số theo SNR\n%s", name, best_epoch, val_table.round(4).to_string())
        class_names = Mixture36Dataset(split="validation", duration=DURATION).class_names
        analysis_files = analyze_predictions(model, loaders["validation"], "validation", name, run.result_dir, class_names)
        test_table = None
        if eval_test:
            test_loader = build_eval_loader("test", train_config["eval_batch_size"], workers)
            test_table = per_snr_table(evaluate(model, test_loader, desc=f"{name} - test"))
            analysis_files += analyze_predictions(model, test_loader, "test", name, run.result_dir, class_names)
            logger.info("%s: TEST, 9 chỉ số theo SNR\n%s", name, test_table.round(4).to_string())
        if RUN_TRAINING:
            checkpoint_path = run.checkpoint_path(name)
            save_model(checkpoint_path, model, epoch=best_epoch, macro_acc=best_score, frcrn=str(FRCRN_PATH.relative_to(ROOT)))
            run.result_dir.mkdir(parents=True, exist_ok=True)
            val_table.to_csv(run.result_dir / f"per_snr_validation_{name}.csv")
            table_files = [f"per_snr_validation_{name}.csv"]
            if test_table is not None:
                test_table.to_csv(run.result_dir / f"per_snr_test_{name}.csv")
                table_files.append(f"per_snr_test_{name}.csv")
            if use_drive:
                for table_file in table_files:
                    DrivePut(fs, remote_dir, table_file, logger)(run.result_dir / table_file)
                for analysis_file in analysis_files:
                    DrivePut(fs, remote_dir, analysis_file.name, logger)(analysis_file)
            pd.DataFrame(history).to_csv(run.result_dir / f"history_{name}.csv", index=False)
        if drive_logfile is not None:
            drive_logfile(run.log_file)
        summary = {"model": name, "best_epoch": best_epoch, "val_macro_acc": best_score,
                   "val_curve": [r["val_macro_acc"] for r in history if r["epoch"] > 0]}
        return summary, (test_table if test_table is not None else val_table)



    return compare_runs, run_experiment, run_name


@app.cell
def _(
    DATASET_ROOT,
    RUNS,
    RUN_TRAINING,
    compare_runs,
    device,
    run_experiment,
    run_name,
    torch,
):
    # (9) Chạy hàng đợi RUNS ở cell (1). RUN_TRAINING = True huấn luyện đầy đủ, False chạy smoke test nhỏ (mỗi nhánh một lần).
    # Chỉ chạy cell này khi đã đồng ý dùng máy. Mỗi nhánh có resume.pt riêng, nên Run all lần sau nối tiếp được thay vì làm lại từ đầu.
    results = []
    for _cfg in RUNS:
        name = run_name(_cfg)
        print(f"dataset: {DATASET_ROOT} | thí nghiệm: {name} | RUN_TRAINING={RUN_TRAINING} | thiết bị: {device}")
        summary, table = run_experiment(name, _cfg)
        results.append(summary)
        print({k: v for k, v in summary.items() if k != "val_curve"})
        torch.cuda.empty_cache()
    comparison = compare_runs(results)
    comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hiệu chỉnh bias theo lớp (post-hoc, sau huấn luyện)

    Student thường dự đoán lệch về một số lớp (Telephone, Flute, Gong... bị dự đoán thừa; Drawer, Hi-hat, Violin, Microwave... bị dự đoán thiếu). Thay vì huấn luyện lại, cộng một bias theo lớp vào logit: `z'_c = z_c + b_c`, rồi `p' = softmax(z')`, `ŷ = argmax p'`. Trọng số, kiến trúc và pipeline đặc trưng giữ nguyên.

    **Bias theo prior.** `b_c = τ · log(q_c / p_c)`, với `q_c = 1/C` (val và test đều 180 clip/lớp) và `p_c` là tỉ lệ mẫu val được student dự đoán là lớp c trước hiệu chỉnh. Lớp dự đoán thừa có `p_c > q_c` nên `b_c < 0`; lớp dự đoán thiếu có `b_c > 0`. `τ ∈ [0, 1]` là cường độ.

    | Chế độ | Áp dụng cho | Khi nào dùng |
    | --- | --- | --- |
    | A (conservative) | chỉ lớp có `p_c/q_c > 1,25` hoặc `< 0,80`, còn lại `b_c = 0` | an toàn hơn khi val nhỏ hoặc nguồn nhiễu val/test khác nhau |
    | B (full) | cả C lớp | khi lệch dự đoán lan rộng |

    **Quy trình (`src/calibration.py`, hàm `run_calibration` ở cell dưới).**

    1. Chạy student trên val, lấy logits, tính baseline (accuracy, macro recall/precision/F1, confusion, tỉ lệ dự đoán/thật từng lớp).
    2. Quét lưới `τ ∈ {0; 0,1; 0,2; 0,3; 0,5; 0,75; 1}` cho cả hai chế độ (không dùng gradient).
    3. Chọn theo thứ tự: macro recall val, rồi macro F1 val, rồi độ lệch dự đoán trung bình `mean_c |n̂_c/n_c − 1|` nhỏ hơn. Ràng buộc: không lớp nào mất recall quá 0,05, trừ khi macro recall tăng ít nhất 0,02.
    4. Chấp nhận nếu trên val macro recall tăng ≥ 0,02 hoặc macro F1 tăng ≥ 0,015, có không quá 2 lớp mất recall quá 0,05, và độ lệch dự đoán không tăng.
    5. Chỉ khi được chấp nhận mới khoá tham số và đánh giá test **một lần**. Không chọn τ, chế độ hay tập lớp bằng test.

    **Biến thể có regularization** (`fit_regularized`): học `b` tự do bằng `min CE(z + b) + λ‖b‖²`, khởi tạo `b = 0`, L-BFGS tối đa 200 bước, `λ ∈ {0,01; 0,1; 1}`. Dùng khi bias theo prior dao động mạnh.

    **Temperature scaling** chia logit cho `T` không đổi thứ tự nên không đổi top-1 và không sửa prior shift; chỉ dùng nếu cần xác suất đáng tin.

    Nếu macro recall không tăng hoặc các cặp nhầm âm sắc (Violin–Clarinet, Chime–Glockenspiel...) không đổi, vấn đề nằm ở biểu diễn: dừng hiệu chỉnh và chuyển sang hard-negative sampling, contrastive loss hay augmentation bảo toàn cao độ. Artifact ghi vào `out_dir`: `calibration_config.json`, `class_bias.json`, `val_metrics_{before,after}.json`, `confusion_matrix_val_{before,after}.csv`, `predicted_count_ratio.csv`.

    **Báo cáo đi kèm (để kết luận đáng tin).**

    - **Ổn định theo SNR và nguồn nhiễu** (`compare_groups`, cần `snr` và `noise_source` trong `val_meta`, `test_meta`): macro recall trước/sau cho từng mức SNR và từng nguồn nhiễu. Nhóm nào giảm quá 0,02 được liệt kê trong `worse`. Bias được fit trên toàn val nên có thể giúp ở SNR cao nhưng hại ở SNR thấp, và con số tổng sẽ che điều đó.
    - **Cặp nhầm ưu tiên** (`pair_confusions`): số lần nhầm hai chiều và tỉ lệ trên tổng mẫu thật của hai lớp, trước/sau, trên val và test (Violin–Clarinet, Chime–Glockenspiel, Saxophone–Harmonica, Saxophone–Trumpet, Clarinet–Trumpet, Acoustic guitar–Electric piano, Fireworks–Gunshot, Keys jangling–Writing, Writing–Computer keyboard, Microwave oven–Gong; khớp tên lớp không phân biệt hoa thường, cặp không khớp ghi `pairs_unresolved`). Bias sửa được lỗi do prior; cặp nhầm gần như không đổi nghĩa là lỗi do biểu diễn, cần hard-negative hoặc contrastive thay vì bias. Cặp nhầm tăng mạnh nghĩa là hiệu chỉnh chỉ dời lỗi sang chỗ khác.
    - **Lớp đảo chiều** (`flipped_classes`): lớp từ dự đoán thiếu (< 0,80) sang thừa (> 1,25) trên test hoặc ngược lại, ghi vào `flipped_classes` của summary.
    - **`run_manifest.json`** (`write_manifest`): git commit, SHA-256 của từng checkpoint, cấu hình đặc trưng, ngày chạy và seed. Dùng để chắc rằng bias fit cho checkpoint nào thì chỉ áp cho checkpoint đó, và để so các lần chạy sau này. Ghi ngay từ đầu, kể cả khi hiệu chỉnh không được chấp nhận.

    **Đã thử trên logits thật của student** (epoch 27): chia validation thành hai nửa theo lớp, fit trên nửa này và đo trên nửa kia (5 hạt giống, 10 lần fit). Macro recall chỉ tăng +0,002 (0,3946 → 0,3966) dù độ lệch dự đoán giảm từ 0,32 xuống 0,26, và cả 10 lần fit đều không đạt tiêu chí chấp nhận. Thiên lệch prior có thật nhưng không phải nguyên nhân chính của khoảng cách với teacher, nên hiệu chỉnh bias không phải đòn bẩy chính cho student này; ưu tiên là cải thiện huấn luyện (ablation KD, hard-negative, ESKD).

    Artifact bổ sung trong `out_dir`: `{val,test}_macro_recall_by_{snr,noise_source}.json`, `confusable_pairs.json`, `run_manifest.json`.
    """)
    return


@app.cell
def _(ROOT, np):
    # (10) Hiệu chỉnh bias theo lớp (post-hoc): fit trên val, khoá, rồi áp dụng MỘT lần lên test. Không đổi trọng số.
    from src import calibration as cal


    def run_calibration(val_logits, val_labels, test_logits, test_labels, class_names, out_dir,
                        val_meta=None, test_meta=None, checkpoints=None, feature_config=None, seed=None):
        """Quét (chế độ A/B) x tau trên val, chọn theo macro recall -> macro F1 -> độ lệch dự đoán, kiểm tra tiêu chí chấp nhận.
        Chỉ khi được chấp nhận mới đánh giá test (một lần) và ghi artifact vào out_dir.
        val_meta / test_meta: dict tuỳ chọn {"snr": mảng, "noise_source": mảng} để báo cáo ổn định theo nhóm.
        checkpoints: {tên: đường dẫn} để băm vào run_manifest.json cùng git commit, feature_config và seed."""
        class_names = list(class_names)
        if checkpoints is not None:
            cal.write_manifest(out_dir, checkpoints, feature_config or {}, seed, ROOT)  # ghi cả khi không được chấp nhận
        rows, base = cal.sweep(val_logits, val_labels)
        chosen = cal.select(rows)
        ok = cal.accepted(base, chosen)
        summary = {"accepted": ok, "mode": chosen["mode"], "tau": chosen["tau"],
                   "val_macro_recall": (base["macro_recall"], chosen["metrics"]["macro_recall"]),
                   "val_macro_f1": (base["macro_f1"], chosen["metrics"]["macro_f1"])}
        if not ok:
            return summary, rows  # không chấp nhận: dừng, chuyển sang cải thiện representation
        bias = chosen["bias"]
        config = cal.CalibrationConfig(mode=chosen["mode"], tau=chosen["tau"], class_names=class_names)
        cal.save_artifacts(out_dir, config, bias, base, chosen["metrics"], class_names)
        test_before, test_after = cal.metrics(test_logits, test_labels), cal.metrics(test_logits, test_labels, bias)  # test chỉ chạy tại đây
        summary["test_macro_recall"] = (test_before["macro_recall"], test_after["macro_recall"])
        summary["test_macro_f1"] = (test_before["macro_f1"], test_after["macro_f1"])
        summary["flipped_classes"] = [class_names[i] for i in cal.flipped_classes(test_before["pred_ratio"], test_after["pred_ratio"])]
        for split, logits, labels, meta in (("val", val_logits, val_labels, val_meta), ("test", test_logits, test_labels, test_meta)):
            for key in ("snr", "noise_source"):
                if meta and meta.get(key) is not None:
                    report = cal.compare_groups(logits, labels, np.asarray(meta[key]), bias)
                    cal.save_report(out_dir, f"{split}_macro_recall_by_{key}", report)
                    summary[f"{split}_worse_{key}"] = report["worse"]
        pairs = {"val_before": cal.pair_confusions(base["confusion"], class_names), "val_after": cal.pair_confusions(chosen["metrics"]["confusion"], class_names),
                 "test_before": cal.pair_confusions(test_before["confusion"], class_names), "test_after": cal.pair_confusions(test_after["confusion"], class_names)}
        cal.save_report(out_dir, "confusable_pairs", pairs)
        summary["pairs_unresolved"] = [p["pair"] for p in pairs["val_before"] if not p["resolved"]]
        return summary, rows


    return


if __name__ == "__main__":
    app.run()
