# Đọc và tái hiện bài báo

## Mục lục
1. Lấy văn bản bài báo
2. Thứ tự đọc
3. Mẫu đầu ra
4. Đối chiếu công thức với mã
5. Giải thích "vì sao tác giả chọn cách này"
6. Ví dụ đã làm trong repo (TGR-MoE)

## 1. Lấy văn bản bài báo

Thứ tự thử, dừng ở cái đầu tiên cho văn bản đọc được:

1. `https://arxiv.org/html/<id>` (bản HTML, có công thức và bảng; chỉ có với bài nộp gần đây).
2. `https://ar5iv.labs.arxiv.org/html/<id>` (bản HTML dựng lại cho bài cũ; đã dùng được cho FRCRN 2206.07293).
3. `https://www.alphaxiv.org/abs/<id>.md` (bản tóm tắt dạng markdown).
4. `https://arxiv.org/abs/<id>` (chỉ abstract, đủ để biết bài nói gì, không đủ để tái hiện).
5. PDF: công cụ tải trang trả về luồng nén nhưng lưu file về máy. Trích văn bản cần `pypdf` hoặc `pymupdf`; môi trường hiện **chưa cài**, hỏi trước khi cài. Nếu không trích được, nói rõ không đọc được thân bài.

Công cụ tải trang tóm tắt qua một mô hình nhỏ, nên con số đọc được vẫn có thể sai. Với số sẽ dùng trong báo cáo (kích thước dữ liệu, SNR, kết quả bảng), hỏi lại đúng câu đó một lần nữa hoặc đối chiếu bản thứ hai.

Khi chỉ có bản tóm tắt, mọi chi tiết cấu trúc lớp, siêu tham số, số ablation đều phải gắn nhãn "chưa kiểm chứng". Một số bài chỉ nói "lightweight router" mà không cho cấu trúc; ghi điều đó vào khoảng trống tái hiện thay vì tự điền.

URL các bài đã dùng của repo nằm ở `doc/paper-url.yaml` (FRCRN, CED). Thêm bài mới vào đó khi người dùng muốn giữ.

## 2. Thứ tự đọc

1. Abstract và hình tổng quan: bài giải quyết vấn đề gì, đo bằng gì.
2. Phần phương pháp: tìm công thức xác định đóng góp (loss, cơ chế định tuyến, kiến trúc).
3. Bảng ablation: cái nào bỏ đi làm kết quả tụt. Đó là cốt lõi.
4. Phụ lục: siêu tham số, lịch huấn luyện, chi tiết bị giấu.
5. Related work: chỉ đọc khi cần biết bài kế thừa từ đâu.

## 3. Mẫu đầu ra

```markdown
# <Tên bài> (<nguồn, năm>)
**Một câu:** bài làm gì và hơn gì.

## Bản đồ đóng góp
| Tuyên bố | Cơ chế | Bằng chứng |
|---|---|---|

## Dữ liệu và giao thức
| Thí nghiệm | Train | Test | Tạo dữ liệu (mix, SNR, reverb) | Tiền xử lý | Metric |
|---|---|---|---|---|---|
- Bài không nói: (độ dài clip, cách chia val, trộn sẵn hay trộn khi train...)

## Cốt lõi và phần phụ
- Cốt lõi: ...
- Phần phụ (mẹo, siêu tham số): ...

## Vì sao tác giả chọn cách này
- Vấn đề -> cơ chế -> nếu bỏ đi thì sao

## Công thức <-> mã
| Công thức | Mã trong repo | Lệch so với bài |
|---|---|---|

## Khoảng trống tái hiện
- Bài không nói: ...

## Áp dụng cho MLNC
- Thử rẻ nhất: ...
```

## 4. Đối chiếu công thức với mã

Với mỗi công thức, kiểm lần lượt:

- **Ký hiệu và chiều:** tensor nào, shape gì, chiều nào được lấy tổng.
- **Hằng số:** nhiệt độ, hệ số cân bằng, epsilon, và nó nằm ở config nào.
- **Chuẩn hoá:** có chia cho batch, cho số token, hay cho số nhóm không.
- **Gradient:** có `detach` / `stopgrad` ở đâu (ví dụ phân phối teacher phải detach).
- **Thời điểm:** áp dụng ở giai đoạn nào, tắt khi nào.

Ví dụ đã kiểm trong repo:

| Công thức | Mã | Ghi chú |
|---|---|---|
| `T² · KL(p_t ‖ softmax(r/T))` | `src/loss/kdmoe.py::router_kl` | `p_t` được tính từ logit teacher đã `detach`, `p_t = softmax(z_t/T) · M`; nhân T² đúng như Hinton |
| `g = ε + (1−ε)·σ(Wh+b)` | `src/model/kdmoe/gate.py::MaskGate` | `ε = 0,05` làm sàn, bias khởi tạo 2,0 nên cổng mở lúc đầu |
| L1 bất đối xứng, phạt thừa ×λ | `src/loss/kdmoe.py::FrontLoss` | `diff > 0` nghĩa là ước lượng nhiễu lớn hơn thật, tức tiếng nói rò vào |
| `L_cons = 1 − cos(e_i, e_j)` | `src/loss/kdmoe.py` (ConsistencyLoss) | cặp là hai mix cùng clip nhiễu, khác tiếng nói hoặc SNR |

## 5. Giải thích "vì sao tác giả chọn cách này"

Dùng ba câu hỏi, trả lời bằng cơ chế chứ không bằng "vì nó tốt hơn":

1. **Vấn đề trước đó là gì?** (ví dụ router chỉ nhận gradient qua expert được chọn.)
2. **Cách này tháo nút thắt đó thế nào?** (teacher router cho giám sát dày cho mọi expert.)
3. **Bỏ nó thì sao?** (xem ablation; không có ablation thì nói là suy đoán.)

Phân biệt "tác giả nói" với "bạn suy ra". Lý do tác giả nêu phải bám văn bản bài báo.

## 6. Ví dụ đã làm: TGR-MoE (CVPR 2026, arXiv 2604.21330)

Dữ kiện đọc được từ bản HTML:

- Router học từ **teacher router** dựng trên đặc trưng trung gian của teacher dày đóng băng (các tầng 8, 10, 12), không phải tầng cuối.
- Loss student: `KL(stopgrad(p_t) ‖ p)`, λ_distill = 5,0. Teacher router chỉ được huấn luyện bằng load-balancing và entropy loss (λ = 0,005).
- Ablation: chưng cất chỉ nửa đầu quá trình train đạt 78,13%, cao hơn giữ suốt (77,81%); dùng đặc trưng tầng cuối chỉ 75,83% (thấp hơn baseline 77,39%).
- Bài **không** nêu cấu trúc lớp của cả hai router, và Eq. 10 không có số hạng cân bằng dù Fig. 2 có. Đó là khoảng trống tái hiện.

Áp vào repo: KD của repo ở mức clip (một vector), router là MLP 512→64→4; điều có thể thử rẻ là lịch chưng cất kiểu nửa đầu (đổi `kd_weight`), vì nó chỉ cần sửa cấu hình.
