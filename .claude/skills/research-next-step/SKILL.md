---
name: research-next-step
description: Đọc tình hình dự án nghiên cứu và đề xuất bước tiếp theo dựa trên bằng chứng có thật. Tự đọc README, cấu hình, log huấn luyện, bảng kết quả, ghi chú và lịch sử git; dựng bức tranh (mục tiêu, đã làm, kết quả, vướng mắc, ràng buộc); tìm nút thắt chính theo thứ tự từ rẻ đến đắt; rồi đưa 2-4 lựa chọn thật sự khác nhau kèm một khuyến nghị. Hãy dùng skill này mỗi khi người dùng hỏi "bước tiếp theo là gì", "giờ nên làm gì", "kết quả thế này thì nên thử gì", "chọn hướng nào", "có nên đổi kiến trúc không", "ưu tiên thí nghiệm nào với GPU có hạn", hoặc vừa dán một bảng kết quả hay log rồi hỏi hướng đi, kể cả khi họ không nhắc tên skill. Use to read the state of a research project and recommend the next step with evidence, cost and quick checks.
---

# research-next-step

Người làm nghiên cứu thường có nhiều hướng đi và ít thời gian GPU. Giá trị của skill này là giúp họ **chọn đúng hướng kế tiếp dựa trên bằng chứng có thật**, thay vì liệt kê mọi thứ có thể làm. Trả lời bằng ngôn ngữ người dùng đang dùng.

Vai: **Researcher** (thu bằng chứng, đề xuất hướng); đầu ra qua bảng kiểm Judge ở `skill-choosing`.

## Bước 1: Thu thập hiện trạng

Ưu tiên tự đọc, chỉ hỏi những gì không thể tìm thấy.

1. Liệt kê file trong workspace, rồi đọc những file mang thông tin trạng thái: README, file cấu hình (yaml/json), log huấn luyện, bảng kết quả (csv/md), ghi chú thí nghiệm, lịch sử git nếu có.
2. Chỉ đọc phần cần thiết của file lớn (đuôi log, dòng tổng kết), không đọc hết.
3. Nếu thiếu mục tiêu, ràng buộc (tài nguyên tính toán, hạn chót) hoặc điều người dùng đã thử, hãy hỏi gọn, tối đa 2-3 câu một lần. Nếu workspace trống, bắt đầu bằng các câu hỏi này.

### Nơi đọc trạng thái trong repo MLNC

| Cần biết | Đọc |
|---|---|
| Mục tiêu, giả thuyết, ablation, tiêu chí chấp nhận | `doc/moe.md` §10, `doc/moe-plan.md` |
| Kết quả đã chạy thật (teacher, student S/L) | `doc/moe-molab-run.md` §6, §9 |
| Kết quả từng lần chạy | `output/result/<exp>-<ts>/result.csv`, `history/*.json`; log ở `output/log/` |
| Bảng so sánh baseline | `output/result/baselines-<ts>/comparison.csv`; bản gộp nhiều lần chạy ở `output/result/baselines-merged-<ngày>/result.csv` (cột `top1_pct`, `macro_f1_pct`, `mAP_pct`, `macro_auc_pct`, `best_epoch`, `epochs_run`) |
| Cấu hình đã dùng | `config/kdmoe_{s,m,l}.yaml`, `config/baselines.yaml` (một bản sao nằm cạnh kết quả: `output/result/<run>/baselines.yaml`) |
| Cách chạy và đọc baseline | README mục "Baseline", `scripts/run_baselines.py`, `notebook/baselines.ipynb.py` |
| Các thay đổi gần đây | `.log/p1.md`, `git log` |
| Ràng buộc tính toán | molab giới hạn 12 giờ có resume; GPU thường RTX 6000 96 GB (`experiment-setting`) |
| Ràng buộc về phạm vi sửa | `rules.MD` mục 0: nếu không được yêu cầu, chỉ tuỳ chỉnh hyperparameters |

Nếu tiến trình đang chạy ở molab, đọc log và `result.csv` thay vì chạy mã trong kernel đang bận.

Lưu ý khi đọc kết quả:

- Một số thư mục trong `output/result/` chỉ có `history/` mà chưa có `result.csv` (lần chạy dở hoặc kết quả nằm ở nơi khác như `output/molab-run/`). Ghi rõ là "không có bảng tổng kết" thay vì suy từ history.
- `output/test/` và các lần chạy `baselines-smoke-*` là smoke test (1 epoch, vài chục clip, CPU), kết quả ở mức ngẫu nhiên. **Không đưa vào so sánh.**
- Bảng gộp baseline có thể trộn nhiều run (cột `run`) với cấu hình train khác nhau; kiểm tra `epochs_run` và `best_epoch` trước khi xếp hạng. `best_epoch` sát `epochs_run` hoặc bằng `epochs` nghĩa là chưa hội tụ, còn `epochs_run` quá bé so với `patience` thường là dừng sớm.

## Bước 2: Dựng bức tranh

Viết ra cho chính mình trước khi phân tích:

- **Mục tiêu:** dự án cần đạt gì, đo bằng chỉ số nào.
- **Đã làm:** các thí nghiệm đã chạy và cấu hình chính.
- **Kết quả:** số liệu hiện tại so với baseline và so với mục tiêu.
- **Vướng mắc:** lỗi, kết quả bất thường, thứ chưa kiểm chứng.
- **Ràng buộc:** dữ liệu, GPU, thời gian, nơi triển khai.

Tách rõ **sự kiện** (đọc được từ file/log) và **suy luận** (bạn đoán). Điều chưa biết thì ghi là chưa biết, không điền bằng số tự nghĩ ra.

## Bước 3: Phân tích

Trả lời câu hỏi "kết quả này thực sự nói lên điều gì". Xét các nguyên nhân phổ biến theo thứ tự **rẻ đến đắt**:

1. **Lỗi dữ liệu hoặc quy trình:** rò rỉ giữa train/test (cùng người nói, cùng nguồn, cùng clip nhiễu), nhãn nhiễu, tiền xử lý không khớp giữa train và eval.
2. **Độ tin cậy của so sánh:** chênh lệch có lớn hơn dao động giữa các seed không, baseline có công bằng không (cùng split, cùng metric, cùng ngân sách huấn luyện).
3. **Dấu hiệu huấn luyện:** overfit, loss không giảm, gradient bất thường.
4. **Lỗi đặc thù kỹ thuật đang dùng:** ví dụ chưng cất tri thức mà teacher quá tự tin, router MoE bị sụp về một expert, teacher-student mà hai bên không cùng điều kiện augmentation.

Mục đích là xác định **nút thắt chính**, không phải liệt kê mọi vấn đề.

Gợi ý cho dự án này (kiểm chứng trước khi tin): so sánh số giữa các bảng chỉ khi cùng split và cùng cách tính (top-1 gộp SNR khác `macro_acc` qua 6 SNR); đọc kết quả theo từng SNR; router acc theo SNR; `L_front` và `gate_mean`/`gate_low` để biết cổng có sụp không.

Khi so KD-MoE với baseline:

- Baseline huấn luyện từ đầu trên cùng log-mel, SpecAugment và optimizer (`config/baselines.yaml`). Kiểm tra KD-MoE cũng dùng cùng dataset (`audioset-40l` hay `mixture-36l`), cùng `duration`/`num_frames` và cùng metric trước khi nói "hơn" hay "kém".
- So theo **ngân sách tham số**: cột `params_m` của baseline với kích thước S/M/L của student (mục tiêu dưới 15M). Hơn một model 27M không có nghĩa là thắng ở 5M.
- Xếp hạng baseline theo `macro_f1` (cách `comparison.csv` xếp) hoặc `top1`; khoảng cách vài phần trăm giữa các hạng gần nhau chưa chắc vượt dao động seed khi mỗi model chỉ chạy một seed (kiểm tra cột `run`/cấu hình `seed` trước khi kết luận).
- Nếu một baseline đứng đầu có `best_epoch` ở sát giới hạn epoch, kết quả có thể còn tăng; nếu `best_epoch` nhỏ và `epochs_run` bằng `best_epoch + patience`, nó đã bão hoà.

## Bước 4: Đề xuất lựa chọn

Đưa ra **2-4 lựa chọn thật sự khác nhau về hướng đi**. Mỗi lựa chọn nêu:

- **Làm gì**, trong một hai câu.
- **Kỳ vọng thu được gì** và dựa trên bằng chứng nào.
- **Chi phí** (thời gian, GPU) và rủi ro.
- **Cách kiểm tra nhanh** trước khi đầu tư lớn (thí nghiệm nhỏ, một batch, một seed).

Sau đó chọn **một khuyến nghị** và giải thích lý do bằng chính bằng chứng ở Bước 2-3. Khi các lựa chọn gần ngang nhau, ưu tiên cái **cho nhiều thông tin nhất với chi phí thấp nhất**, vì nó giúp quyết định các bước sau chính xác hơn.

## Định dạng trả lời

Trả lời ngắn trong chat, theo khung sau:

```
**Hiện trạng**: 2-4 câu, gồm số liệu chính so với baseline/mục tiêu.
**Điều đáng chú ý**: nút thắt hoặc phát hiện quan trọng nhất (nêu rõ là sự kiện hay suy luận).
**Các lựa chọn**:
1. ... (ưu / nhược / cách thử nhanh)
2. ...
**Khuyến nghị**: chọn gì và vì sao.
**Cần xác nhận**: những điều chưa biết có thể đổi khuyến nghị (nếu có).
```

Có thể dùng danh sách khi so sánh các lựa chọn, nhưng giữ tổng thể gọn, đọc một lần là nắm được.

## Nguyên tắc

- Không đưa kết luận chắc chắn khi dữ liệu còn mỏng. Nói rõ mức độ tin cậy.
- Không khuyên thay đổi lớn khi chưa loại trừ lỗi đơn giản (rò rỉ dữ liệu, bug tiền xử lý).
- Nếu kết quả tốt bất thường, nghi ngờ trước khi ăn mừng.
- Tôn trọng ràng buộc tài nguyên của người dùng: đề xuất cần khả thi với GPU và thời gian họ có.
- Tôn trọng `rules.MD`: mặc định các lựa chọn là chỉnh hyperparameters hoặc thêm thí nghiệm. Đề xuất đổi kiến trúc, thêm loss hay thêm file thì gắn nhãn "cần người dùng cho phép sửa mã".
- **Không sửa hay xoá file của người dùng khi chỉ được yêu cầu phân tích.**
- Không chạy huấn luyện hay đánh giá để "xem thử"; nêu lệnh, nơi chạy và chi phí rồi chờ cho phép (`skill-choosing`).

## Chuyển tiếp sang skill khác

Sau khi người dùng chọn hướng:

| Việc tiếp | Skill |
|---|---|
| Lập kế hoạch thí nghiệm, seed, metric, ước lượng GPU | `experiment-setting` |
| Cài đặt thay đổi, nạp checkpoint, gỡ lỗi | `implement-setting` |
| Hiểu cơ chế hoặc đọc bài báo liên quan | `research-platform` |
| Viết lại kết quả (kể cả âm tính) | `soft-skill` |
| Yêu cầu rộng, không rõ làm gì tiếp ngoài "bước tiếp theo" | `skill-choosing` |

`anthropic-skills:research-next-steps` (bản chung, không biết cấu trúc repo) cùng chức năng với skill này. Trong repo MLNC dùng skill này; bản chung chỉ dành cho dự án khác.
