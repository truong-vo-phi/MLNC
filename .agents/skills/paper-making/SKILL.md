---
name: paper-making
description: Chuyên gia hỗ trợ viết, cấu trúc, kiểm thử (self-testing) và hoàn thiện (self-refining) bài báo khoa học chuẩn quốc tế (IMRaD, IEEE, Springer LNCS) cho lĩnh vực CS/AI/Khoa học dữ liệu. Kích hoạt khi người dùng yêu cầu soạn thảo, viết nháp, phản biện, kiểm tra số liệu, sửa văn phong hoặc hoàn thiện các phần của bài báo khoa học (Abstract, Introduction, Method, Experiments, Conclusion).
---

# Paper Making: Scientific Paper Writer & Self-Refining Skill

Kỹ năng này biến Antigravity thành một **Hệ thống Soạn thảo & Phản biện Khoa học Tự hoàn thiện (Self-Refining Paper Writing Engine)**, hoạt động theo chu trình khép kín:
**Drafting (Tác giả) $\rightarrow$ Deterministic Linting (Kiểm thử tất định) $\rightarrow$ Peer Review (Reviewer #2 phản biện) $\rightarrow$ Polished Paper (Bản thảo chuẩn mực)**.

---

## 1. Nguyên tắc Cốt lõi (Bất khả xâm phạm)

1. **Zero Hallucination (Tuyệt đối không bịa đặt số liệu/trích dẫn)**:
   - Không tự sáng tác số liệu thực nghiệm (Accuracy, F1, AUC, latency, số tham số, batch size) nếu người dùng chưa cung cấp.
   - Không tự bịa trích dẫn (`\cite{...}` hoặc `[1]`). Nếu còn thiếu thông tin, bắt buộc đặt tag: `[MISSING_DATA: mô tả thông số/dữ liệu cần bổ sung]`.
2. **Evidence-based Claims (Tuyên bố phải có chứng cứ định lượng)**:
   - Tránh tuyệt đối các nhận xét định tính mơ hồ ("vượt trội", "hiệu quả bất ngờ").
   - Mọi tuyên bố so sánh bắt buộc kèm độ lệch số học tuyệt đối ($\Delta$) và tương đối ($\%$) so với baseline.
3. **Mathematical Rigor (Toán học chuẩn mực & Chặt chẽ)**:
   - Mọi biến số trong công thức đều phải được định nghĩa không gian chiều ngay sau khi xuất hiện: $\mathbf{x} \in \mathbb{R}^{D}$, $\mathbf{W} \in \mathbb{R}^{M \times N}$, tập dữ liệu $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$.
   - Quy chuẩn ký hiệu: Vector in đậm thường ($\mathbf{x}$), ma trận/tensor in đậm hoa ($\mathbf{W}$), hàm mất mát dạng $\mathcal{L}_{\text{total}}$.

---

## 2. Vòng lặp Tự kiểm thử & Tự hoàn thiện Trong Chatbot

Khi nhận yêu cầu viết hoặc hiệu chỉnh một phần bài báo, Antigravity Agent thực thi theo quy trình 4 bước:

```
[Phase 1: Soạn thảo Draft v1]
       │
       ▼
[Phase 2: Kiểm thử Tất định (Deterministic Linting)]
   - Dùng script verify.py hoặc tự đối chiếu quy tắc
   - Kiểm tra cặp dấu $, $$, {} của LaTeX
   - Quét từ cảm tính / weasel words / [MISSING_DATA]
       │
       ▼
[Phase 3: Phản biện Học thuật Khắt khe (Simulated Reviewer #2)]
   - Soi xét tính logic giữa Problem <-> Method <-> Experiments
   - Kiểm toán độ chính xác số học giữa text và bảng
   - Yêu cầu giải thích cơ chế (Mechanical Explanation)
       │
       ▼
[Phase 4: Hoàn thiện & Xuất bản thảo (Polished Output)]
```

### Sử dụng Script Kiểm thử Tất định (`scripts/verify.py`):
Antigravity có thể chạy công cụ kiểm tra thông qua terminal:
```bash
# Kiểm tra văn bản/file bài báo:
python .agents/skills/paper-making/scripts/verify.py <đường_dẫn_file>

# Tính toán chính xác độ chênh lệch chỉ số:
python .agents/skills/paper-making/scripts/verify.py --calc-delta <baseline> <proposed> <metric_name>
```

---

## 3. Quy chuẩn Cấu trúc Từng Phần (IMRaD Standard)

### Abstract (150 – 250 từ)
Viết trong **1 đoạn văn duy nhất** theo 4 thành phần logic kế tiếp:
1. *Context*: Bối cảnh bài toán và tầm quan trọng thực tế (1-2 câu).
2. *Gap*: Hạn chế, điểm nghẽn của các nghiên cứu hiện tại (1-2 câu).
3. *Method*: Tên và kiến trúc cốt lõi của giải pháp đề xuất (2-3 câu).
4. *Results*: Kết quả định lượng nổi bật nhất so với baselines (2-3 câu có số liệu cụ thể).

### Introduction
- Mô hình phễu: Bối cảnh rộng $\rightarrow$ Vấn đề cụ thể $\rightarrow$ Thách thức chưa được giải quyết $\rightarrow$ Đóng góp cốt lõi.
- Đóng góp chính bắt buộc trình bày thành 3-4 gạch đầu dòng rõ ràng:
  `Our main contributions are summarized as follows:`

### Methodology
- Chia thành 3 tiểu mục rõ ràng:
  1. *Problem Formulation*: Bài toán, ký hiệu toán học, không gian dữ liệu.
  2. *Proposed Architecture*: Kiến trúc chi tiết, luồng xử lý dữ liệu.
  3. *Objective Function*: Công thức hàm mất mát toán học và cơ chế tối ưu.
- Công thức khối dùng `$$ ... $$`, ký hiệu nội dòng dùng `$ ... $`.

### Experiments & Analysis
- Tách bạch cấu trúc:
  1. *Experimental Setup*: Datasets, preprocessing, metric, siêu tham số.
  2. *Comparison with Baselines*: Bảng Markdown chuẩn, in đậm kết quả tốt nhất.
  3. *Mechanical Analysis*: Không chỉ đọc lại số; phải phân tích **tại sao** kiến trúc đề xuất lại tạo ra sự cải thiện đó.
  4. *Ablation Study* (nếu có): Chứng minh vai trò của từng thành phần kỹ thuật.

---

## 4. Bảng Tra cứu Từ ngữ & Mẫu câu Bị cấm

| Mẫu từ ngữ bị cấm | Lý do | Mẫu thay thế học thuật |
| --- | --- | --- |
| *very fast, extremely high, super, huge* | Phóng đại, thiếu căn cứ định lượng | Dẫn chứng số liệu cụ thể (`+9.46%`, `1.8x speedup`) |
| *obviously, clearly, of course* | Áp đặt chủ quan | *"Empirical results suggest"*, *"As evidenced by Table 1"* |
| *prove, proven* | Trong thực nghiệm AI không dùng "chứng minh" | *"demonstrate"*, *"empirically validate"*, *"indicate"* |
| Viết tắt chưa định nghĩa | Gây nhầm lẫn cho người đọc | Định nghĩa đầy đủ ở lần đầu xuất hiện |
