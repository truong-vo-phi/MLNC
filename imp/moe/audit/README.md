# Audit học thuật `imp/moe` — hướng dẫn đọc

Thư mục này chứa bộ kiểm tra học thuật cho pipeline `imp/moe`, đối chiếu với bài báo
**BEATs: Audio Pre-Training with Acoustic Tokenizers** (arXiv:2212.09058v1) và code chính thức
của Microsoft (`microsoft/unilm/beats`), cùng các thành phần riêng của repo (CED-Tiny, FRCRN,
KD, routing, TTA, calibration).

Mục tiêu: tránh 3 loại lỗi thường gặp trước hội đồng:

1. **Gán sai phương pháp** — gọi một thành phần là "của BEATs" khi BEATs không làm vậy.
2. **So sánh sai** — đem số của dataset 36 lớp ra so với số AudioSet/ESC-50 của paper.
3. **Không truy xuất được** — con số trong README không map được tới một lần chạy, checkpoint,
   manifest và raw prediction.

## Các file

| File | Nội dung |
| :--- | :--- |
| `01_audit_matrix.csv` | 98 tiêu chí: nguồn chuẩn, cách kiểm, bằng chứng trong repo, kết quả, mức độ, tác động, khuyến nghị. Đây là file chính. |
| `02_summary.csv` | Tổng hợp trạng thái/mức độ, 5 kết luận lớn, top 10 rủi ro. |
| `03_priority_actions.csv` | 19 việc ưu tiên, có thứ tự, đầu ra cần có, ước lượng và phụ thuộc. |
| `build_audit_csv.py` | Script sinh 3 file CSV; chạy lại khi sửa đánh giá. |

Chạy lại:

```bash
python imp/moe/audit/build_audit_csv.py
```

## Cách đọc cột `Kết quả`

| Giá trị | Nghĩa |
| :--- | :--- |
| `ĐẠT` | Khớp paper/code, có bằng chứng trực tiếp. |
| `ĐẠT CÓ ĐIỀU CHỈNH` | Khớp nhưng có adaptation hợp lý; phải ghi rõ trong báo cáo. |
| `MỘT PHẦN` | Đúng một phần; còn điểm lệch hoặc thiếu. |
| `KHÔNG ĐẠT` | Sai/thiếu so với yêu cầu học thuật. |
| `CHƯA XÁC MINH` | Không có checkpoint/artifact để kiểm tra; không kết luận đúng/sai. |
| `KHÔNG ÁP DỤNG` | Ngoài phạm vi folder hoặc paper. |

## Cách đọc cột `Mức bằng chứng`

| Mã | Nguồn |
| :--- | :--- |
| `P` | Paper text (section/table/equation). |
| `C` | Code chính thức của tác giả (BEATs). |
| `R` | Code/config trong repo `imp/moe`. |
| `A` | Artifact: manifest, summary.json, benchmark CSV, log. |
| `I` | Suy luận từ bằng chứng (chưa chạy thực nghiệm). |
| `U` | Chưa xác minh (không có file/checkpoint). |

Nguyên tắc: **không dùng `I` hoặc `U` để khẳng định một kết quả là đúng/sai**. Nếu một dòng
là `CHƯA XÁC MINH`, cần bổ sung artifact trước khi đưa vào báo cáo chính thức.

## 5 kết luận lớn

1. **Kiến trúc BEATs cơ bản là đúng**: 12 layers, 768 dim, 3072 FFN, patch 16, 128 Mel,
   25/10 ms, mean pooling + linear head. Tuy nhiên paper ghi **8 attention heads**, còn code
   chính thức và repo dùng **12 heads** — phải ghi footnote, không được trích paper cho H=12.
2. **Chưa phải end-to-end**: `ChampionPipeline` không chạy FRCRN; eval đọc cache `waveform`.
   Cần ghi rõ "FRCRN offline precomputed" hoặc implement on-the-fly.
3. **Kết quả 63.58% là oracle routing**: `evaluate_pipeline` truyền `target_snr_db` thật vào
   `predict_batch`; `EnergySNRRouter` không được exercise. Cần báo cáo cả oracle và estimated.
4. **Selection leakage**: weights/temperatures và high-expert được tune/chọn trên chính
   validation 3.064 mẫu rồi báo cáo trên đó. Cần test set 3.253 mẫu độc lập.
5. **Metric/target chưa đúng**: `mAP_macro = 70.08%` là trung bình per-SNR, không phải macro
   trên 36 lớp; `mAP_overall = 69.57%` < 0.70. `target_achieved: true` chưa hợp lệ theo định
   nghĩa thông thường.

## 3 kịch bản trước hội đồng

### Kịch bản A — bảo vệ nguyên trạng (không khuyến nghị)

Rủi ro: 13 mục `CRITICAL`, trong đó có oracle SNR, selection leakage, metric mapping,
artifact hardcode, FRCRN không có trong pipeline, checkpoint provenance. Hội đồng có thể yêu
cầu chạy lại và sẽ thấy số không tái lập.

### Kịch bản B — sửa tối thiểu (2-3 ngày)

Ưu tiên: `H01`, `E03/F05`, `G03/G04/G11`, `F03`, `B06`, `C04/C05`, `E02`, `A03`, `I02`.
Kết quả: minh bạch được số liệu và định nghĩa, nhưng vẫn còn gap FRCRN/checkpoint/test set.
Phù hợp nếu hội đồng chỉ yêu cầu giải trình, không yêu cầu tái lập độc lập.

### Kịch bản C — sửa đầy đủ (1-2 tuần)

Làm toàn bộ `03_priority_actions.csv`: test set độc lập, selection split, FRCRN/cache metadata,
checkpoint manifest, numerical equivalence test, CI/seed, regenerated benchmark CSV.
Đây là mức cần thiết nếu muốn bảo vệ ở chuẩn công bố hoặc chuyển sang bài báo.

## Giới hạn của audit này

- Chưa có checkpoint `/marimo/...` và cache trên máy local, nên các mục `B13`, `B14`, `D03`,
  `D04`, `E09`, `E10` ở trạng thái `CHƯA XÁC MINH` hoặc dựa trên code/runner, không dựa trên
  log chạy thực tế.
- Các kết luận `KHÔNG ĐẠT` về code là kiểm tra tĩnh (đọc code/config), không phải chạy
  end-to-end.
- Không khẳng định arXiv ID của CED/FRCRN trong audit này; cần tra đúng trước khi trích.
- Các con số per-SNR được đếm trực tiếp từ `manifests.zip` (validation: 3.064; test: 3.253).

## Cách dùng trước khi trình bày

1. Mở `01_audit_matrix.csv`, lọc `Mức độ = CRITICAL` và `HIGH` — đây là phần hội đồng sẽ hỏi.
2. Với mỗi dòng `CHƯA XÁC MINH`, bổ sung artifact hoặc hạ mức độ trong báo cáo.
3. Dùng `03_priority_actions.csv` làm checklist tiến độ; đánh dấu `Done` khi có đầu ra.
4. Khi viết báo cáo, mỗi con số phải có cột `run_id`/`hash`/`manifest` đi kèm.
