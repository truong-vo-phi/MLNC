# Viết kỹ thuật: mẫu và danh sách kiểm

## Mục lục
1. Báo cáo một thí nghiệm
2. Trình bày kết quả âm tính
3. README
4. Mô tả đóng góp (abstract, contribution)
5. Danh sách kiểm trước khi gửi
6. Lỗi thường gặp
7. Báo cáo capstone (LaTeX)

## 1. Báo cáo một thí nghiệm

```markdown
## <ID>: <tên ngắn>

**Kết luận.** <một hoặc hai câu: có hay không, bao nhiêu, ở điều kiện nào>

**Câu hỏi và giả thuyết.** ...

**Thiết lập.**
| Mục | Giá trị |
|---|---|
| Cấu hình | `config/kdmoe_s.yaml` + thay đổi: ... |
| Dữ liệu | mix-dataset, 30 240 / 6 480 / 6 480 |
| Seed | 42, 43, 44 |
| Mã | commit `abc1234` |

**Kết quả.**
| Cấu hình | −5 | 0 | 5 | 10 | 15 | 20 | macro |
|---|---:|---:|---:|---:|---:|---:|---:|

**Phân tích.** ...

**Giới hạn.** Chưa kiểm: ...

**Bước tiếp theo.** Thí nghiệm rẻ nhất để tăng niềm tin: ...
```

Quy ước: giá trị đo ghi kèm đơn vị (phần trăm hay phân số); cột in đậm chỉ khi chênh lệch vượt dao động seed.

## 2. Trình bày kết quả âm tính

Kết quả âm tính hữu ích khi nó **hẹp phạm vi câu hỏi**. Cách viết:

1. **Nêu đúng điều đã làm và đã thấy**, không đổi mục tiêu sau khi thấy kết quả.
2. **Đưa số và điều kiện** như kết quả dương.
3. **Nêu các giải thích có thể, gắn nhãn chưa kiểm chứng,** và nói thí nghiệm nào sẽ phân biệt.
4. **Nêu giá trị của nó:** "loại trừ X như nguyên nhân" hoặc "cho thấy nút thắt không nằm ở Y".
5. **Quyết định:** giữ nguyên, đổi, hay dừng, và vì sao.

Ví dụ khung (kết quả trong repo): *Tăng dung lượng student S → L (≈ 11 lần tham số) chỉ tăng top-1 khoảng 2,7 điểm, trong khi teacher cao hơn S khoảng 23 điểm. Điều này cho thấy dung lượng student không phải nút thắt chính (một seed, chưa kiểm chứng với nhiều seed). Giả thuyết còn lại: kiến thức truyền từ teacher quá hẹp (KD chỉ vào router) hoặc thiếu pretrain. Thí nghiệm rẻ nhất: thêm KD 36 lớp.*

Tránh: "mô hình không hiệu quả" (không nói gì), đổ lỗi dữ liệu mà không kiểm, hoặc bỏ không báo.

## 3. README

```markdown
# <Tên dự án>
<Một đoạn: bài toán, cách tiếp cận, điểm khác biệt.>

## Cấu trúc
<cây thư mục, mỗi dòng một vai trò>

## Cài đặt
```bash
python -m venv .venv
pip install -e ".[dev]"
```

## Dữ liệu
<bảng khoá cấu hình, nội dung, nguồn, cách tải>

## Chạy
1. ...
2. ...

## Kiểm thử
```bash
pytest -m "not data"
```
```

Mỗi bước chạy cho **lệnh đúng** và **đầu ra mong đợi**. Không để token hay đường dẫn máy trong README.

## 4. Mô tả đóng góp (abstract, contribution)

Cấu trúc abstract ngắn (khoảng 150 từ):

1. Bối cảnh và vấn đề (1 câu).
2. Khoảng trống hoặc khó khăn (1 câu).
3. Cách tiếp cận (1 đến 2 câu).
4. Kết quả chính có số (1 đến 2 câu).
5. Ý nghĩa hoặc giới hạn (1 câu).

Đóng góp viết ở dạng **kiểm chứng được**: "Chúng tôi đề xuất X, giảm Y từ A xuống B ở SNR 15–20 dB" thay vì "chúng tôi nghiên cứu X".

## 5. Danh sách kiểm trước khi gửi

- [ ] Câu đầu mỗi mục nói điều quan trọng nhất.
- [ ] Mỗi con số có điều kiện (cấu hình, split, seed).
- [ ] Có mục giới hạn và điều chưa kiểm.
- [ ] Thuật ngữ thống nhất (dùng một tên cho một khái niệm).
- [ ] Ký hiệu được định nghĩa ở lần đầu; bảng và hình có chú thích tự đứng được.
- [ ] Tên file, hàm, lệnh dùng đúng định dạng mã.
- [ ] Không có token, đường dẫn máy, thông tin riêng.
- [ ] Đọc lại bằng mắt của người chưa biết dự án.

## 6. Lỗi thường gặp

| Lỗi | Sửa |
|---|---|
| Chôn kết luận ở cuối | Đưa lên đầu |
| Hứa hẹn quá mức ("vượt trội") | Nêu số và điều kiện |
| Số không điều kiện | Thêm cấu hình, split, seed |
| Nhiều tên cho một thứ | Chọn một tên, định nghĩa một lần |
| Câu dài nhiều mệnh đề | Tách thành hai câu |
| Giấu kết quả xấu | Viết như kết quả dương, kèm hướng điều tra |

## 7. Báo cáo capstone (LaTeX)

Khung ở `latex/capstone/` (mẫu FPTU, `report`, IEEEtran). `main.tex` ghép các chương trong `chapters/`; mỗi chương hiện chỉ có tiêu đề và ghi chú hướng dẫn bọc trong `\btext{...}` / `\rtext{...}`. Khi viết, thay ghi chú bằng nội dung, đừng để sót.

Lấy nội dung từ nguồn có sẵn, không viết lại từ trí nhớ:

| Chương | Nguồn trong repo |
|---|---|
| I. Project Introduction | `doc/moe.md` §1 (phạm vi, giả định), README |
| III. Methodology, mục dữ liệu | `doc/moe.md` §2, `scripts/load_dataset.py::mix_at_snr` |
| III. Methodology, mục mô hình | `doc/moe.md` §4–7, `src/model/kdmoe/`, `src/loss/kdmoe.py` |
| III. Experimental Settings | `config/kdmoe_{s,m,l}.yaml`, `config/baselines.yaml`, `doc/moe.md` §8 |
| III. Evaluation Metrics | `src/metrics.py`, `experiment-setting` mục chọn chỉ số |
| IV. Results and Discussion | `doc/moe-molab-run.md` §6, §9; `output/result/baselines-merged-*/result.csv`; `doc/moe.md` §10 |
| V. System Design | README, `scripts/`, `doc/moe.md` §9 |
| References | `doc/paper-url.yaml` → mục BibTeX trong `latex/capstone/references.bib` |

Quy ước:

- Bảng kết quả theo khuôn ở mục 1 (cột theo SNR + macro), số đi kèm cấu hình, split, seed.
- Mỗi bài trích dẫn có một mục BibTeX; trích bằng `\cite{key}`. Không tự điền tác giả, năm, venue khi chưa đọc trang `abs`.
- Báo cáo viết tiếng Anh: dùng `references/academic-english.md`.
- Biên dịch là chạy lệnh trên máy: nêu lệnh (`latexmk -pdf main.tex` trong `latex/capstone/`) và chờ người dùng cho phép. Preamble dùng `\patchcmd` nhưng dòng nạp `etoolbox` đang bị comment, có thể lỗi khi build.
