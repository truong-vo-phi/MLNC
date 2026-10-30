---
name: soft-skill
description: Kỹ năng mềm cho nghiên cứu trong repo MLNC. Gồm viết kỹ thuật (báo cáo, README, bài báo, ghi chú kết quả, kể cả khi kết quả âm tính); quản lý thời gian và phạm vi (làm bản nhỏ chạy được trước rồi mới mở rộng, smoke test, chia mốc, quyết định cắt gì); tư duy phản biện (nghi ngờ kết quả quá tốt, kiểm tra trước khi tin, tìm rò rỉ và lỗi đo); và tiếng Anh học thuật để đọc tài liệu và viết bài. Hãy dùng skill này mỗi khi người dùng viết hoặc sửa báo cáo, README, tài liệu, abstract, mô tả đóng góp, trình bày một kết quả hay một thí nghiệm thất bại, hỏi nên làm gì trước, bị quá tải phạm vi, nghi ngờ một con số, hoặc cần dịch, viết, đọc tiếng Anh học thuật, kể cả khi họ không nhắc tên skill. Use for technical writing, scoping and prioritisation, critical thinking about results, and academic English.
---

# soft-skill

Bốn kỹ năng này quyết định liệu công việc kỹ thuật tốt có được hiểu, tin và dùng lại hay không. Trả lời bằng tiếng Việt; khi nói về tiếng Anh học thuật thì đưa ví dụ tiếng Anh kèm giải nghĩa.

Vai: **Professional** (chế độ A, B, D) và **Judge** (chế độ C là nguồn của bảng kiểm trong `skill-choosing`).

## Chọn chế độ

| Người dùng nói gì | Chế độ | Đọc thêm |
|---|---|---|
| "viết báo cáo", "README", "trình bày kết quả", "kết quả không như mong đợi", "viết chương capstone" | A. Viết kỹ thuật | `references/writing.md` (§7 cho `latex/capstone/`) |
| "làm gì trước", "không kịp", "mở rộng thế nào" | B. Thời gian và phạm vi | mục bên dưới |
| "kết quả này có đáng tin không", "quá tốt", "kiểm tra" | C. Tư duy phản biện | mục bên dưới |
| "viết tiếng Anh", "câu này nghe lạ", "đọc bài báo này" | D. Tiếng Anh học thuật | `references/academic-english.md` |

## A. Viết kỹ thuật

Nguyên tắc: **người đọc hôm nay không có ngữ cảnh của bạn.** Một tài liệu tốt trả lời được ba câu mà không phải hỏi lại: nó làm gì, chạy thế nào, tin được đến mức nào.

- **Bắt đầu bằng kết luận**, rồi mới đến lý do và chi tiết. Câu đầu của mỗi mục nói điều quan trọng nhất.
- **Mọi con số đi kèm điều kiện:** cấu hình, seed, tập dữ liệu, đơn vị. Ví dụ "top-1 31,1% (student S, val, 1 seed, best @ epoch 25)".
- **Kết quả âm tính là kết quả.** Viết đúng như đã xảy ra, kèm giả thuyết nguyên nhân gắn nhãn "chưa kiểm chứng". Ví dụ đã có trong repo: "Router acc 68,8%, chưa đạt 0,95, giữ inference weighted" (`doc/moe-molab-run.md` §6) là cách viết đúng: nêu số, nêu ngưỡng, nêu quyết định.
- **Phân biệt ba mức chắc chắn:** đo được, suy ra từ mã, giả thuyết.
- **Ghi cái chưa làm.** Một mục "Giới hạn" ngắn tốn ít công và tăng độ tin cậy.
- **README** cần: mục đích một đoạn, cấu trúc, cài đặt, cách chạy tối thiểu, định dạng dữ liệu, cách chạy test. Lệnh để trong khối `bash`, mỗi lệnh một khối.

Mẫu báo cáo thí nghiệm và README nằm trong `references/writing.md`.

## B. Quản lý thời gian và phạm vi

Nguyên tắc: **một bản nhỏ chạy được từ đầu đến cuối luôn có giá trị hơn một bản đầy đủ chạy được một nửa.**

1. **Lát cắt dọc nhỏ nhất.** Đi hết pipeline với dữ liệu rất ít và mô hình nhỏ: nạp dữ liệu, một bước huấn luyện, đánh giá, ghi kết quả. Repo có sẵn `smoke` (1 epoch, vài clip) cho việc này.
2. **Rồi mới mở rộng**, mỗi lần một chiều: dữ liệu, rồi dung lượng, rồi số seed, rồi ablation.
3. **Cổng quyết định trước khi tốn tiền.** Ví dụ: chạy teacher xong, kiểm router acc ở 15–20 dB ≫ 25% rồi mới chạy student (`doc/moe.md` §10.3). Cổng rẻ ngăn tuần lãng phí.
4. **Ước lượng trước khi chạy** (skill `experiment-setting`, `references/compute-budget.md`), và ghi lại số thực sau khi chạy.
5. **Cắt phạm vi có chủ đích.** Với mỗi việc, hỏi: nếu bỏ nó thì kết luận chính có đổi không? Nếu không, để vào danh sách "sau".
6. **Làm đúng yêu cầu** (`rules.MD`): không thêm tính năng ngoài yêu cầu; chỉ tuỳ chỉnh hyperparameters nếu không được yêu cầu khác.

Khi người dùng quá tải, đề xuất bằng một danh sách ba cột ngắn: **làm ngay**, **làm nếu còn thời gian**, **bỏ**, kèm một dòng lý do cho từng mục.

## C. Tư duy phản biện

Nguyên tắc: **kết quả bất ngờ là giả thuyết về lỗi cho đến khi chứng minh ngược lại.** Càng đẹp càng phải kiểm.

Thứ tự kiểm khi một con số quá tốt, hoặc tốt hơn mong đợi:

1. **Rò rỉ dữ liệu:** split theo nhóm (clip nhiễu, người nói), giao tập id rỗng, chuẩn hoá chỉ tính trên train, probe cross-fit.
2. **Lỗi đo:** metric đúng chiều (macro hay micro), đúng split, không tính nhãn đệm, SNR trong manifest đúng.
3. **Đường tắt:** mô hình đọc được thứ không nên đọc (độ lớn, thời lượng, id, thống kê chuẩn hoá của mixture).
4. **So sánh không công bằng:** baseline thiếu epoch, lr khác, seed may rủi.
5. **Dao động seed:** chạy thêm seed trước khi tin hiệu ứng nhỏ.
6. **Đối chứng tầm thường:** mô hình ngẫu nhiên, đoán lớp đa số, đặc trưng đơn giản. Ngẫu nhiên 36 lớp là 2,8% (`doc/moe-molab-run.md`).

Với kết quả xấu, kiểm theo chiều ngược: mã (shape, gradient), dữ liệu (nhãn lệch), cấu hình (lr). Chi tiết ở `implement-setting/references/debugging.md`.

Khi nhận xét về kết quả của chính mình, nói theo mẫu: **"Số liệu cho thấy X. Nó nhất quán với giả thuyết Y, nhưng cũng nhất quán với Z. Thí nghiệm rẻ nhất để phân biệt là W."**

## D. Tiếng Anh học thuật

Hai việc riêng: **đọc** (hiểu nhanh bài báo) và **viết** (rõ, chính xác, không quá tự tin). Mẫu câu, từ nối, cách viết thận trọng (hedging), lỗi thường gặp của người nói tiếng Việt và cách đọc nhanh bài báo nằm trong `references/academic-english.md`. Khi sửa câu cho người dùng, giữ ý của họ, nêu lý do ngắn, đưa một bản sửa.

## Nguyên tắc

- Không tô hồng và không bi quan hoá: viết đúng mức chắc chắn của bằng chứng.
- Giữ ngắn. Một đoạn chính xác hơn ba đoạn mơ hồ.
- Khi người dùng nhờ viết hộ, giữ giọng của họ và các số liệu đúng như đã cho; không tự bịa số.
