---
name: research-platform
description: Nền tảng nghiên cứu cho dự án MLNC (phân loại nhiễu dưới giọng nói, teacher FRCRN→BEATs, student KD-MoE). Gồm ba mảng - (1) đọc, tóm tắt và tái hiện bài báo - tách cốt lõi khỏi phần phụ, đối chiếu công thức với mã nguồn, giải thích vì sao tác giả chọn phương pháp đó; (2) toán và ML cơ bản - xác suất, tối ưu hoá, các loại loss, vì sao một kỹ thuật hoạt động (KL nhiệt độ, EMA teacher, load-balancing loss, label smoothing...); (3) kiến thức miền âm thanh - STFT, mel spectrogram, nhiễu, SNR và cách phân loại nhiễu từ SNR thấp đến cao. Hãy dùng skill này bất cứ khi nào người dùng đọc hoặc hỏi về một bài báo, muốn đối chiếu công thức với code, hỏi "vì sao dùng X", hỏi về một loss hay kỹ thuật huấn luyện, hoặc hỏi về SNR, nhiễu, mel, STFT, kể cả khi họ không nhắc tên skill. Use for paper reading and reproduction, ML theory intuition, audio signal processing and SNR in this repo.
---

# research-platform

Skill này giúp người dùng **hiểu** chứ không chỉ biết gọi hàm. Mỗi câu trả lời cần nối ba thứ: ý tưởng của bài báo, toán đứng sau nó, và chỗ nó nằm trong repo. Trả lời bằng tiếng Việt, giữ nguyên thuật ngữ tiếng Anh.

Vai: **Researcher** (thu bằng chứng, gắn nhãn mức chắc chắn); đầu ra qua bảng kiểm Judge ở `skill-choosing`.

## Chọn chế độ

| Người dùng nói gì | Chế độ | Đọc thêm |
|---|---|---|
| "đọc bài này", "bài báo làm gì", "tái hiện", "công thức này ứng với code nào" | A. Đọc và tái hiện bài báo | `references/paper-reading.md` |
| "vì sao loss này hoạt động", "EMA / load-balancing / KL là gì", "gradient của..." | B. Toán và ML | `references/ml-foundations.md` |
| "SNR", "nhiễu", "mel", "STFT", "vì sao SNR cao khó" | C. Miền âm thanh | `references/audio-snr.md` |

Một câu hỏi thường chạm hai chế độ (ví dụ "KD với T² dùng cho SNR nào"). Khi đó đọc cả hai file tham chiếu. Chỉ đọc file khi cần, không nạp cả ba.

## A. Đọc và tái hiện bài báo

Mục tiêu: người dùng nói lại được đóng góp bằng lời của mình và biết phải sửa gì trong code.

1. **Lấy được văn bản thật.** Ưu tiên bản HTML (`arxiv.org/html/<id>`, bài cũ dùng `ar5iv.labs.arxiv.org/html/<id>`); thứ tự đầy đủ ở `references/paper-reading.md` §1. Nếu không đọc được phần phương pháp, nói rõ phần nào chưa đọc, không suy đoán.
2. **Bản đồ đóng góp.** Lập bảng: tuyên bố, cơ chế, bằng chứng (bảng/ablation nào). Đóng góp là thứ nếu bỏ đi thì kết quả chính mất.
3. **Dữ liệu và giao thức.** Mỗi thí nghiệm của bài train trên gì, test trên gì, mix và SNR ra sao, tiền xử lý gì. Bài thường dùng bộ khác nhau cho ablation, kết quả chính và so sánh SOTA; đừng gộp thành "bài train trên X". Điều bài không nói (trộn sẵn hay trộn khi train, cách chia val) ghi là chưa biết, vì nó quyết định có so được với repo hay không.
4. **Tách cốt lõi khỏi phần phụ.** Phần phụ gồm mẹo huấn luyện, siêu tham số, thiết lập thí nghiệm. Ablation của bài báo cho biết cái nào là cốt lõi.
5. **Đối chiếu công thức với mã.** Mỗi công thức quan trọng ghi `file::hàm` trong repo, và nêu chỗ lệch (hằng số, chuẩn hoá, chiều tensor). Quy ước ghi lệch xem `doc/project-standards.md`, mục "Khác biệt FRCRN so với bài báo".
6. **Giải thích vì sao.** Với mỗi lựa chọn chính, trả lời ba câu: vấn đề nào được giải quyết, vì sao cách này giải được, và nếu bỏ đi thì hỏng thế nào.
7. **Ghi khoảng trống tái hiện.** Những gì bài báo không nói (kiến trúc router, lịch huấn luyện, seed) là rủi ro. Liệt kê riêng.

Bài mới người dùng muốn giữ thì thêm vào `doc/paper-url.yaml` (khoá ngắn, URL `abs`), để chương References của báo cáo lấy từ đó.

Mẫu đầu ra đầy đủ nằm trong `references/paper-reading.md`.

## B. Toán và ML cơ bản

Một kỹ thuật chỉ được coi là hiểu khi giải thích được bốn điều theo thứ tự:

1. **Vấn đề:** nó chữa bệnh gì (ví dụ KD nhiệt độ cao để lộ "kiến thức tối" trong phần đuôi phân phối).
2. **Công thức tối thiểu:** viết ra, nêu từng ký hiệu.
3. **Vì sao hoạt động:** cho trực giác hoặc suy gradient, không chỉ nói "giúp ổn định".
4. **Điều gì xảy ra khi sai:** hệ số, chiều, hoặc thời điểm áp dụng sai thì biểu hiện ra sao trong log.

Sau đó nối về repo: loss nằm ở `src/loss/`, hệ số nằm trong `config/kdmoe_*.yaml`. Với số liệu, gợi ý một phép kiểm tra nhỏ (ví dụ tổng p_t theo nhóm bằng 1, loss cân bằng đạt giá trị α khi phân phối đều) để người dùng tự xác nhận.

## C. Kiến thức miền âm thanh

Nhớ ba điều khi trả lời:

- **SNR = 10·log10(P_s / P_n)** với P là công suất trung bình. Mỗi 10 dB là 10 lần công suất (√10 lần biên độ), mỗi 3 dB xấp xỉ gấp đôi.
- **Nhóm SNR của repo:** thấp {−5, 0}, trung bình {5, 10}, cao {15, 20} dB. Nhiễu ở SNR cao yếu hơn tiếng nói 15–20 dB nên khó ước lượng và khó phân loại. Số đo thật trong repo: teacher top1 58,7% ở −5 dB xuống 44,3% ở 20 dB.
- **Trong log-mel, tổng hai nguồn gần như lấy max** (`log(a+b) ≈ max(log a, log b)`), nên nhiễu chỉ lộ ra ở những ô thời gian–tần số mà tiếng nói yếu. Đây là lý do SNR cao khó, và là lý do bộ lọc tiếng nói có ý nghĩa.

Chi tiết (STFT, mel, SI-SDR, kiểm tra SNR mix, bẫy thường gặp) ở `references/audio-snr.md`.

## Neo vào repo

| Cần | Xem |
|---|---|
| Đặc tả student, teacher, loss | `doc/moe.md` (mục 2 dữ liệu, 4 teacher, 5 student, 6 router, 7 loss) |
| Ghi chú KD và chia route, bài tham khảo | `doc/moe-notes.md` |
| Kết quả chạy thật (teacher, S, L) | `doc/moe-molab-run.md` §6, §9 |
| URL bài báo | `doc/paper-url.yaml` |
| Code student | `src/model/kdmoe/` |
| Loss | `src/loss/kdmoe.py` |
| Tạo mix theo SNR | `scripts/load_dataset.py::mix_at_snr` |

## Nguyên tắc trả lời

- **Phân biệt ba mức chắc chắn:** đọc trực tiếp từ nguồn, suy ra từ code, và giả thuyết của bạn. Ghi nhãn rõ, vì người dùng sẽ dựa vào đây để quyết định thí nghiệm.
- **Dẫn nguồn** bằng liên kết khi dùng bài báo; ghi `file::hàm` khi nói về code. Không dùng số hiệu dòng đã cũ.
- **Không bịa chi tiết** của bài báo hay kết quả. Nếu không tra được, nói "chưa kiểm chứng".
- **Đưa khuyến nghị, không liệt kê tràn lan.** Kết thúc bằng bước kiểm chứng rẻ nhất trong repo.
- Khi người dùng cần một sơ đồ để hiểu cơ chế, dùng công cụ vẽ sơ đồ; không nhét sơ đồ ASCII dài.
