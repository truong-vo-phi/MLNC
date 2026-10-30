# Thiết kế thí nghiệm, thống kê, chỉ số, phân tích lỗi

## Mục lục
1. Khung một thí nghiệm
2. Ablation: cách lập và đọc
3. Seed và thống kê
4. Chọn chỉ số
5. Tập test độc lập
6. Phân tích lỗi theo nhóm
7. Báo cáo bảng kết quả

## 1. Khung một thí nghiệm

```markdown
### <ID>: <tên ngắn>
- Câu hỏi: ...
- Giả thuyết: nếu <thay đổi> thì <chỉ số> <tăng/giảm> ở <điều kiện>.
- Bị bác bỏ nếu: <ngưỡng cụ thể>.
- Thay đổi so với A0: duy nhất một dòng cấu hình.
- Giữ cố định: dữ liệu, split, seed, số epoch, lr.
- Chỉ số: ...
- Chi phí ước tính: <giờ GPU>, <VRAM>.
```

Điều kiện bác bỏ cụ thể ngăn việc diễn giải kết quả theo hướng mình muốn.

## 2. Ablation: cách lập và đọc

- **Loại bỏ** (remove) cho biết thành phần có cần không; **thay thế** (swap) cho biết cách làm cụ thể có tốt hơn đối chứng đơn giản không (ví dụ ViT thuần A8 là đối chứng tối thiểu).
- Cắt từng thành phần nhưng **giữ ngân sách**: cùng số bước, cùng lr; nếu bỏ thành phần làm số tham số đổi, cân nhắc so sánh cả ở số tham số bằng nhau.
- Đọc kết quả theo từng SNR, không chỉ trung bình: một thành phần có thể giúp ở 0 dB và hại ở 20 dB, trung bình triệt tiêu.
- **Hiệu ứng tương tác:** hai thành phần giúp riêng lẻ chưa chắc cộng dồn; chạy cả hai khi muốn khẳng định.

Chạy theo mức ưu tiên: (a) thí nghiệm chỉ đổi cấu hình, (b) đổi một loss, (c) đổi kiến trúc. Càng đắt càng sau.

## 3. Seed và thống kê

| Tình huống | Làm gì |
|---|---|
| Khác biệt lớn (≫ 2 điểm) và nhất quán theo SNR | 1 seed có thể đủ cho khảo sát sơ bộ, ghi rõ |
| Khác biệt 0,5 đến 2 điểm | 3 seed, báo trung bình ± độ lệch chuẩn |
| Khác biệt < độ lệch chuẩn | Không kết luận hơn kém; thêm seed hoặc coi là ngang nhau |
| Chọn giữa nhiều cấu hình sát nhau | Chọn cấu hình nhỏ nhất trong 1 điểm so với tốt nhất (quy ước `doc/moe.md` §8.7) |

Thực hành:

- Cùng bộ seed cho mọi cấu hình, để so sánh theo cặp (paired).
- Báo độ lệch chuẩn mẫu (n−1); với 3 seed nó rất thô, nên đừng suy diễn khoảng tin cậy hẹp.
- Seed ảnh hưởng ba thứ riêng: khởi tạo, thứ tự dữ liệu, và (nếu có) chia split; ghi seed chia split cố định và **không** đổi giữa các so sánh.
- Với kiểm định chính thức trên cùng tập test, dùng bootstrap theo clip nhiễu (nhóm), không theo mẫu, vì các mix cùng clip nhiễu không độc lập.

## 4. Chọn chỉ số

| Chỉ số | Định nghĩa ngắn | Dùng khi | Hạn chế |
|---|---|---|---|
| Accuracy | tỉ lệ đúng | Lớp cân bằng | Đánh lừa khi lớp lệch |
| Balanced accuracy | trung bình recall mỗi lớp | Lớp lệch | Bỏ qua precision |
| Macro-F1 | trung bình F1 mỗi lớp | Cần cả precision và recall | Nhạy với lớp hiếm |
| Micro-F1 | F1 gộp mọi mẫu | Bằng accuracy với đơn nhãn | Bị lớp lớn chi phối |
| mAP | trung bình AP mỗi lớp | Xếp hạng, đa nhãn | Cần điểm số hiệu chỉnh tốt |
| Macro-AUC | trung bình AUC mỗi lớp | Không phụ thuộc ngưỡng | Lạc quan khi nhiều lớp âm |
| Top-k | đúng nếu nhãn nằm trong k điểm cao nhất | Lớp dễ nhầm | Khó so sánh giữa số lớp |
| EER | điểm FAR = FRR | Xác minh, phát hiện | Không áp dụng cho nhiều lớp trực tiếp |
| ECE | sai lệch giữa độ tin cậy và độ đúng | Cần hiệu chỉnh xác suất (KD, nhiệt độ) | Phụ thuộc cách chia bin |

Quy ước của repo (`src/metrics.py::ClassificationMetrics`): chín số gồm top-1, top-3, balanced accuracy, macro-F1, micro-F1, mAP, macro-AUC và các đại lượng đi kèm; `macro_acc` (trung bình balanced accuracy qua SNR) dùng chọn checkpoint giai đoạn 2; `front_score = −L_front` dùng ở giai đoạn 1.

Chọn **một chỉ số chính** trước khi chạy, ghi trong kế hoạch; các chỉ số khác để chẩn đoán.

## 5. Tập test độc lập

- Chia theo **nhóm** (clip nhiễu, người nói), không theo mẫu; kiểm giao tập rỗng.
- Val dùng để chọn cấu hình và dừng sớm; test chỉ chạy khi đã chốt.
- Nếu đã nhìn test để quyết định, nói rõ và coi nó như val.
- Không đưa test vào chuẩn hoá, probe, hay hiệu chỉnh nhiệt độ.
- Báo cáo test kèm SNR, nhóm và số mẫu mỗi ô.

## 6. Phân tích lỗi theo nhóm

Quy trình đề xuất sau mỗi kết quả chính:

1. **Theo SNR:** accuracy mỗi mốc; xác định mốc yếu nhất.
2. **Theo nhóm expert (E0–E3):** accuracy lớp, và router acc theo nhóm.
3. **Ma trận nhầm lẫn:** `confusion_matrix` và `top_confusions` có sẵn trong `src/metrics.py`; xem 10 cặp nhầm nhiều nhất.
4. **Nhầm trong nhóm hay giữa nhóm:** nếu nhầm giữa nhóm thì router/teacher; nếu trong nhóm thì expert hoặc đặc trưng.
5. **Theo độ khó:** SNR × lớp; lớp nào rơi mạnh ở 15–20 dB.
6. **Mẫu sai cụ thể:** nghe vài mẫu sai để kiểm nhãn và nhiễu có thật như nhãn.

```python
for snr in sorted(set(snrs)):
    m = snrs == snr
    per_snr[snr] = ClassificationMetrics(num_classes=36).update(y[m], p[m]).compute()
```

Ghi lại giả thuyết rút ra, rồi kiểm bằng thí nghiệm nhỏ thay vì kết luận từ ma trận.

## 7. Báo cáo bảng kết quả

- Một bảng cho kết quả chính: hàng là cấu hình, cột là SNR và macro; trung bình ± độ lệch chuẩn.
- In đậm giá trị tốt nhất chỉ khi chênh lệch vượt dao động seed.
- Đơn vị rõ ràng (phần trăm hay phân số); số chữ số thập phân nhất quán.
- Dòng cuối ghi số tham số, FLOP hoặc độ trễ (`doc/moe.md` §10.1).
