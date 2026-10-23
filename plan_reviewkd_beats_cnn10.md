# Kế hoạch: Teacher BEATs – Student FRCRN→PANN-CNN10 với ReviewKD

Nguồn đối chiếu: `latex/beat-reviewkd/main.tex`, `src/model/reviewkd/*`, `src/train/reviewkd_trainer.py`, `config/reviewkd.yaml`.

## 1. Pipeline mục tiêu so với mã hiện có

| Thành phần | Bạn muốn | Mã hiện có | Việc cần làm |
|---|---|---|---|
| Teacher | noise → BEATs (nạp pretrain) | mixture → FRCRN (đóng băng) → n̂ = x − ŝ → chuẩn hóa RMS 0.07925 → BEATs pretrain (lớp 3,6,9,12) | Thêm chế độ teacher nhận **noise thật** (có sẵn trong dataset) thay vì n̂ ước lượng |
| Student | mix → FRCRN (fine-tune, pretrain) → CNN10 | mix → log-mel → CNN10 (không có FRCRN) | **Thiếu**: thêm FRCRN vào đầu student (xem 2.2) |
| KD | ReviewKD | ReviewModule (ABF) + HCL, 4 cặp stage↔lớp, đã chạy được | Giữ nguyên |
| Đầu ra | logit 36 | logits (B, 36) | Giữ nguyên |
| Đánh giá | từng mốc SNR + trung bình | `snr_scores`: acc/balanced/mAP theo từng SNR (−5…20) + `macro_acc`, `mAP_macro` | Giữ nguyên |

Điểm cần chú ý: hiện teacher dùng **FRCRN → n̂**, còn bạn mô tả teacher nhận **noise**. Nếu dùng noise thật làm đầu vào teacher thì đây là "privileged information" (lúc suy luận không có noise), hợp lý cho KD; nếu dùng n̂ thì teacher cũng bị giới hạn bởi chất lượng FRCRN. Cần chọn một (câu hỏi mở 2).

## 2. Các bước triển khai

### 2.1 Teacher
1. Thêm tùy chọn `teacher.input: noise_gt | noise_est` (mặc định `noise_gt`). `noise_gt`: batch cần trả thêm `noise` (kiểm tra `load_dataset.py` có trả không; nếu không thì thêm, có SNR/peak-scale nhất quán với mixture).
2. Giữ chuẩn hóa RMS (0.07925) và `Fbank` đúng thống kê BEATs.
3. Biến thể pretrain: `beats_checkpoint: null` → khởi tạo ngẫu nhiên (xem mục 3).

### 2.2 Student
1. Thêm `FRCRNStudentFrontend`: mix → FRCRN (nạp `pretrained/frcrn_best.pt`, **fine-tune** trước) → ŝ hoặc n̂ → log-mel → CNN10.
2. Quyết định đầu ra FRCRN đưa vào CNN10 (câu hỏi mở 3): vì nhiệm vụ là phân loại **nhiễu**, nên dùng n̂ = x − ŝ (ĐÃ có kết quả SI-SDR 13.5 dB cho "đầu ra trực tiếp" trong log KD-MoE; cần xác nhận lại ý nghĩa target).
3. Fine-tune FRCRN trước (giai đoạn 1), rồi KD (giai đoạn 2). Ở giai đoạn 2: đóng băng FRCRN trước, mở khóa sau N epoch với lr nhỏ (lr_FRCRN ≈ 0.1× lr CNN10) — so sánh trong ablation.
4. Checkpoint phải lưu cả FRCRN lẫn CNN10 nếu FRCRN được cập nhật (hiện `checkpoint_module` chỉ lưu student CNN).

### 2.3 Huấn luyện (giữ cấu hình đã có)
Loss = CE(label smoothing 0.1) + λ·w(e)·Σ HCL(O_i, F_t^i), λ=1, warm-up λ 5 epoch, AdamW lr 3e-4, wd 0.05, 100 epoch, patience 15. Lưới stage 4 = 8×24 khớp BEATs (đã kiểm tra bằng `check_grids`).

### 2.4 Đánh giá
Chạy test độc lập trên 6 mốc SNR, báo cáo từng mốc và trung bình. Ghi `teacher_s_per_step`, `peak_vram_gb` như trainer hiện tại.

## 3. Câu hỏi "không nạp pretrain thì sao" – thiết kế ablation

Dự đoán (chưa đo): BEATs khởi tạo ngẫu nhiên không cho đặc trưng có nghĩa → KD ≈ ép student khớp mục tiêu nhiễu ngẫu nhiên → không hơn, thậm chí kém hơn baseline CE. Vì BEATs pretrain là thứ mang tri thức từ AudioSet 2M, bỏ nó gần như mất nguồn tri thức duy nhất của teacher. Cần kiểm chứng bằng thí nghiệm, đổi một biến mỗi lần:

| ID | Teacher | Student | λ | Mục đích |
|---|---|---|---|---|
| A0 | – | CNN10 (mix→log-mel) | 0 | baseline CE |
| A1 | – | FRCRN(ft)→CNN10 | 0 | tác dụng của FRCRN riêng |
| B1 | BEATs pretrain, noise_gt | FRCRN(ft)→CNN10 | 1 | **cấu hình chính** |
| B2 | BEATs **không pretrain** (random, đóng băng) | như B1 | 1 | trả lời câu hỏi của bạn |
| B3 | BEATs không pretrain, huấn luyện cùng student | như B1 | 1 | biến thể: teacher học từ đầu |
| B4 | BEATs pretrain, n̂ ước lượng (pipeline cũ) | như B1 | 1 | noise thật vs n̂ |
| C | B1 với λ ∈ {0.5, 2}, lớp teacher khác | – | – | độ nhạy (nếu còn tài nguyên) |

Ghi chú B2: đóng băng teacher random là kiểm chứng "KD có thật sự dùng tri thức pretrain hay chỉ là regularizer". Nên dùng ≥3 seed vì chênh lệch dự kiến nhỏ.

## 4. Tham khảo cần đọc (để đối chiếu công thức, chưa tải/đọc trong phiên này)
- ReviewKD: Chen et al., *Distilling Knowledge via Knowledge Review*, CVPR 2021 (ABF + HCL).
- BEATs: Chen et al., *BEATs: Audio Pre-Training with Acoustic Tokenizers*, ICML 2023.
- PANNs: Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks*, 2020 (CNN10).
- FRCRN: Zhao et al., *FRCRN: Boosting Feature Representation using Frequency Recurrence*, ICASSP 2022.
- Repo gốc ReviewKD (dvlab-research/ReviewKD) để đối chiếu `nearest` vs `area`.

## 5. Quyết định đã chốt
1. **Đánh giá** theo từng mốc SNR (−5, 0, 5, 10, 15, 20) và trung bình; không gộp nhóm thấp/cao. Chỉ số gốc: balanced accuracy (`balanced_snr_*`, trung bình là `macro_acc`, cũng dùng chọn checkpoint).
2. **Teacher** nhận nhiễu thật (`noise_gt`); B4 (n̂ ước lượng) giữ làm ablation.
3. **Student** dùng n̂ = x − ŝ từ FRCRN.
4. **FRCRN** fine-tune trước, rồi **đóng băng** trong giai đoạn KD → checkpoint chỉ cần lưu CNN10 + FRCRN đã đóng băng (không đổi cơ chế lưu hiện tại, nhưng pipeline suy luận phải nạp cả FRCRN). Mục 2.2 bước 3–4 vì thế chỉ còn là tùy chọn ablation.

Việc tiếp theo: (a) kiểm tra dataset có trả `noise` thật; (b) cài `FRCRNStudentFrontend` + `teacher.input`; (c) đã có sẵn chỉ số theo từng SNR trong `snr_scores`; (d) smoke test 1 batch trước khi xin chạy thật.

## 6. Rủi ro
- Các điểm "chưa kiểm chứng" ghi trong `main.tex` còn nguyên: lớp BEATs 3/6/9/12, mức RMS 0.07925 so với dữ liệu pretrain, dropout ở cuối mỗi khối CNN10 trước khi vào ReviewKD.
- VRAM/thời gian: thêm FRCRN trong student làm tăng chi phí mỗi bước; cần ước lượng trước khi chạy (GPU 96GB) — chưa chạy gì trên local.
