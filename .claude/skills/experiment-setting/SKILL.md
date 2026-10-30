---
name: experiment-setting
description: Thiết kế, theo dõi và đánh giá thí nghiệm học máy trong repo MLNC (phân loại nhiễu dưới giọng nói, KD-MoE). Bao gồm thiết kế thí nghiệm khoa học (baseline rõ ràng, ablation từng thành phần, đổi một biến mỗi lần, nhiều seed khi kết quả sát nhau); theo dõi cấu hình, phiên bản dữ liệu và kết quả (output/ theo RunPaths, tuỳ chọn MLflow hoặc Weights & Biases); chọn đúng chỉ số (accuracy, balanced accuracy, F1, mAP, AUC, EER), đánh giá trên tập test độc lập và phân tích lỗi theo nhóm và theo SNR; và quản lý tài nguyên tính toán (ước lượng thời gian và bộ nhớ trước khi chạy, thường là một GPU RTX 6000 96 GB VRAM). Hãy dùng skill này mỗi khi người dùng lập kế hoạch ablation, so sánh hai cấu hình, hỏi chạy mấy seed, chọn metric, báo cáo kết quả, phân tích confusion, hoặc ước lượng tốn bao lâu và bao nhiêu VRAM, kể cả khi họ không nhắc tên skill. Use for experiment design, tracking, metrics, error analysis and compute budgeting.
---

# experiment-setting

Một thí nghiệm tốt trả lời **một câu hỏi** bằng **một thay đổi**, với bằng chứng đủ để tin. Trả lời bằng tiếng Việt, giữ thuật ngữ tiếng Anh.

Vai: **Professional** (thiết kế và ước lượng thí nghiệm); đầu ra qua bảng kiểm Judge ở `skill-choosing`.

## Bốn mảng

| Việc | Đọc |
|---|---|
| Thiết kế thí nghiệm, ablation, seed, thống kê, chọn metric, phân tích lỗi | `references/design-and-metrics.md` |
| Ước lượng thời gian và VRAM, kế hoạch chạy trên RTX 6000 96 GB | `references/compute-budget.md` |
| Theo dõi cấu hình và kết quả | mục bên dưới |

## Thiết kế: năm điều bất di bất dịch

1. **Có baseline rõ ràng và chạy lại được.** Trong repo là "A0" (student S). Mọi so sánh quy về A0 cùng seed, cùng dữ liệu, cùng ngân sách huấn luyện.
2. **Đổi một biến mỗi lần.** Nếu hai thứ đổi cùng lúc thì không biết cái nào tạo ra khác biệt. Muốn thử tổ hợp, chạy từng thành phần riêng trước.
3. **Chạy nhiều seed khi kết quả sát nhau.** Quy ước của repo: 3 seed, báo trung bình ± độ lệch chuẩn. Chênh lệch nhỏ hơn độ lệch chuẩn thì chưa kết luận được.
4. **Chọn trên val, báo trên test, một lần.** Test độc lập chỉ chạm khi đã chốt cấu hình.
5. **Báo cả kết quả âm.** Ablation không cải thiện vẫn là thông tin (xem skill `soft-skill`).

Mỗi thí nghiệm cần viết trước ba dòng: **câu hỏi**, **giả thuyết** và **điều gì làm giả thuyết sai**. Không có dòng thứ ba thì kết quả nào cũng "giải thích được" sau đó.

## Ablation của repo

`doc/moe.md` §10.2 định nghĩa A0 đến A11 (bỏ cổng, chỉ attention, chỉ BiGRU, bỏ L_front, bỏ KL, thêm L_route, ViT thuần, S/M/L, top-1, attention pooling). Dùng bảng đó làm nguồn gốc; thêm dòng mới theo cùng khuôn: ID, thay đổi, câu hỏi.

Tiêu chí chấp nhận trong §10.3 (ví dụ A0 không tệ hơn A5 ở macro SNR) là các **cổng**: nếu không đạt, dừng và xem lại nguyên nhân trước khi chạy tiếp.

## Theo dõi thí nghiệm

Repo ghi bằng `RunPaths` (`src/utils/paths.py`):

```
output/log/<exp>-<ts>.log
output/checkpoint/<exp>-<ts>/<model>/last_best_checkpoint.pt
output/result/<exp>-<ts>/result.csv  + history/*.json
output/figure/<exp>-<ts>/*.png
```

Mỗi lần chạy phải ghi lại tối thiểu:

| Mục | Ghi gì |
|---|---|
| Cấu hình | bản yaml đã dùng, `dataclasses.asdict(config)` |
| Dữ liệu | đường dẫn dataset, số mẫu mỗi split, hash manifest, seed chia split và seed sinh mix, phiên bản cache teacher |
| Mã | commit hash (`git rev-parse HEAD`), phiên bản `torch`/`torchaudio` |
| Seed | seed huấn luyện |
| Kết quả | metric mỗi epoch, metric cuối theo SNR và theo nhóm |

**MLflow / Weights & Biases chưa được dùng trong repo.** Nếu người dùng muốn thêm, đề xuất cách tối thiểu: một hàm log duy nhất (`log_params` từ cấu hình, `log_metrics` mỗi epoch, `log_artifact` cho `result.csv`) đặt trong `src/utils/`, gọi từ `Trainer.end_epoch`; không đổi định dạng `output/`. Hỏi trước vì nó thêm phụ thuộc vào `requirements.txt`.

## Chọn chỉ số

Tóm tắt (chi tiết ở `references/design-and-metrics.md`):

| Chỉ số | Dùng khi | Chú ý ở repo |
|---|---|---|
| top-1 accuracy | Lớp cân bằng | Mix được cân bằng theo nhãn và SNR |
| balanced accuracy | Lớp lệch | `macro_acc` chọn checkpoint là trung bình của balanced accuracy qua 6 SNR |
| macro-F1 | Quan tâm cả precision và recall từng lớp | `ClassificationMetrics` |
| mAP, macro-AUC | Xếp hạng, đa nhãn, không phụ thuộc ngưỡng | Chuẩn của AudioSet |
| top-3 | Các lớp dễ nhầm hợp lý | Ít thông tin hơn khi có nhóm |
| EER | Bài toán xác minh (speaker verification) | Không dùng cho phân loại 36 lớp |

Luôn báo **theo từng SNR** (−5, 0, 5, 10, 15, 20) và macro trung bình; số gộp che mất vùng yếu (15–20 dB).

## Ước lượng tài nguyên

Làm trước khi chạy: (1) đếm tham số và bộ nhớ trạng thái, (2) đo 50 bước đầu để lấy giây mỗi bước và đỉnh VRAM, (3) nhân ra tổng thời gian. Công thức, bảng và quy trình trong `references/compute-budget.md`. Đừng tin ước lượng thô nếu chưa đo: `doc/moe.md` §8.5 ghi rõ "chưa đo".

## Nguyên tắc

- Một con số không có cấu hình, seed và dữ liệu đi kèm thì không dùng được.
- Khi kết quả quá tốt hoặc quá bất ngờ, kiểm tra rò rỉ trước (xem `implement-setting/references/debugging.md`).
- Ghi lại **thí nghiệm không chạy được hoặc chưa chạy**, không để người đọc tưởng đã làm.
