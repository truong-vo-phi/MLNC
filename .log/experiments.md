# Nhật ký thí nghiệm: student nhỏ cho teacher BEATs trên Mixture-36l (n̂ từ cache FRCRN)

Mục tiêu: val acc hoặc val mAP đạt 0,6 đến 0,7; sau đó giảm số tham số và giảm chênh lệch train/val.
Quy tắc: val không tăng quanh 5 epoch thì đổi phương pháp. Chỉ dùng validation, chưa động tới test.
Nơi chạy: molab (`/marimo`), dữ liệu n̂ đọc từ `/marimo/cache/nhat`, mọi run dùng cùng teacher `pretrained/beat-ft/best.pt`.

Cách ghi: mỗi lần thử một mục. `val` là macro_acc (= balanced accuracy) trên validation, `gap` là chênh lệch train và val.
Từ E4 trở đi, train acc đo ở chế độ eval trên 2.000 clip train cố định; các run trước đó chỉ có train acc lúc huấn luyện (bị nhiễu bởi dropout, mixup), dùng so gần đúng.

## Mốc tham chiếu

| Mốc | Tham số | Top-1 / macro_acc | mAP | Ghi chú |
|---|---|---|---|---|
| Teacher BEATs 12 lớp (`best.pt`) trên n̂ cache | khoảng 90M | 0,694 | 0,766 | toàn bộ validation, 6.480 clip. Theo SNR: −5 dB 0,755; 0 dB 0,738; 5 dB 0,713; 10 dB 0,694; 15 dB 0,664; 20 dB 0,600 |
| PANNs-CNN10 (baseline cũ, output/result/baselines-merged) | 4,97M | 0,4488 | 0,4478 | tốt nhất trong 16 baseline cũ |

## Các lần thử

### T1. DeiT-Tiny + ReviewKD (4 khối vào lớp 3, 6, 9, 12 của BEATs)
- Cấu hình: `config/reviewkd_deit.yaml`, DeiT-Tiny khởi tạo ImageNet, 5,43M tham số, ReviewKD λ=1, warmup 2 epoch, n̂ từ cache chuẩn hoá RMS 0,05.
- Run: `reviewkd-deit-2026-10-26-03-00-00`. Dừng thủ công sau epoch 5.
- Kết quả:

| Epoch | Train acc | Val |
|---|---|---|
| 1 | 0,364 | 0,339 |
| 2 | 0,640 | 0,388 |
| 3 | 0,798 | 0,403 |
| 4 | 0,863 | 0,393 |
| 5 | 0,900 | 0,381 |

- Kết luận: overfit nhanh, val đỉnh 0,403 ở epoch 3 rồi giảm. Chênh lệch train/val 0,52 ở epoch 5.

### T2. DeiT-Tiny + block-wise KD (mở loss từng khối ở epoch 1, 4, 7, 10)
- Cấu hình như T1, thêm `--blockwise-unlock 1,4,7,10`. Run `reviewkd-deit-2026-10-26-05-00-00`. Dừng thủ công sau epoch 7.
- Kết quả:

| Epoch | Train acc | Val | mAP |
|---|---|---|---|
| 1 | 0,363 | 0,357 | 0,381 |
| 2 | 0,634 | 0,370 | 0,385 |
| 3 | 0,792 | 0,370 | 0,368 |
| 4 | 0,859 | 0,380 | 0,372 |
| 5 | 0,896 | 0,368 | 0,346 |
| 6 | 0,920 | 0,363 | 0,347 |
| 7 | 0,939 | 0,372 | 0,352 |

- Kết luận: không tốt hơn T1, vẫn overfit giống hệt (train 0,94, val 0,37). Block-wise (cách mở dần khối, do mình tự hiểu từ tóm tắt bài báo, chưa đọc bài gốc) không giảm overfit. Bỏ hướng chưng cất đặc trưng cho DeiT; chưa làm q/k/v và logit KD cho DeiT vì không có dấu hiệu đáng làm.

### E2. BEATs cắt 6 lớp, khởi tạo từ teacher, CE + KD logit (không mixup)
- Script `scripts/run_beats_trunc.py`, 47,8M tham số, `--layers 6`, lr 5e-5, KD logit nhiệt độ 4 trọng số 1, label smoothing 0,1. Tên run `e2-beats6L-kd1`.
- Dừng thủ công sau epoch 11 để nhường GPU cho E3 (không lưu `best.pt`). Chọn theo macro_acc (bản script cũ, chưa có tiêu chí gap).
- Kết quả (epoch 4 bị sót khi ghi nhận):

| Epoch | Train acc | Val | mAP |
|---|---|---|---|
| 1 | 0,259 | 0,399 | 0,426 |
| 2 | 0,625 | 0,494 | 0,532 |
| 3 | 0,831 | 0,525 | 0,561 |
| 5 | 0,963 | 0,532 | 0,586 |
| 6 | 0,979 | 0,547 | 0,596 |
| 7 | 0,987 | 0,549 | 0,595 |
| 8 | 0,991 | 0,546 | 0,599 |
| 9 | 0,993 | 0,550 | 0,602 |
| 10 | 0,994 | 0,554 | 0,604 |
| 11 | 0,9955 | 0,557 | 0,610 |

- Kết luận: mAP vượt 0,6 từ epoch 9, nhưng train 0,99 so với val 0,56 (chênh khoảng 0,44).

### E3. Như E2 thêm mixup 0,5 (kết quả tốt nhất cho tới giờ)
- `--mixup 0.5`, 47,8M tham số, 20 epoch chạy hết, `best.pt` ở `/marimo/output/beats-trunc/e3-beats6L-mix0.5/`. Chọn theo macro_acc (bản script cũ). Train acc dưới đây là lúc huấn luyện (có mixup), không so trực tiếp với `train_acc_eval`.
- Kết quả:

| Epoch | Train acc | Val | mAP |
|---|---|---|---|
| 1 | 0,235 | 0,398 | 0,427 |
| 2 | 0,568 | 0,500 | 0,543 |
| 3 | 0,749 | 0,519 | 0,563 |
| 4 | 0,842 | 0,537 | 0,593 |
| 5 | 0,885 | 0,549 | 0,601 |
| 8 | 0,922 | 0,558 | 0,618 |
| 10 | 0,932 | 0,567 | 0,618 |
| 12 | 0,941 | 0,573 | 0,628 |
| **15** | 0,945 | **0,577** | **0,630** |
| 20 | 0,941 | 0,573 | 0,628 |

- Kết luận: tốt hơn E2 ở mọi epoch và chênh lệch nhỏ hơn rõ (train 0,94 thay vì 0,99 ở cuối). Val tốt nhất 0,5767 (epoch 15), mAP 0,630. mAP đạt mục tiêu 0,6 từ epoch 5, macro_acc chưa đạt 0,6 (còn thiếu khoảng 0,02).

### E4. BEATs cắt 4 lớp + mixup 0,5 (34M tham số)
- Dùng bản script có tiêu chí gap. Dừng thủ công sau epoch 4 vì điểm giảm 3 epoch liên tiếp.

| Epoch | Train acc (eval) | Val | mAP | Chênh lệch | Điểm |
|---|---|---|---|---|---|
| 1 | 0,337 | 0,258 | 0,290 | 0,079 | 0,219 |
| 2 | 0,582 | 0,394 | 0,411 | 0,188 | 0,300 |
| 3 | 0,752 | 0,420 | 0,448 | 0,332 | 0,254 |
| 4 | 0,862 | 0,438 | 0,462 | 0,424 | 0,226 |

- Kết luận: cắt từ 6 xuống 4 lớp làm val thấp hơn E3 khoảng 0,10 ở cùng epoch và chênh lệch mở rộng nhanh. Không đáng theo.

### E5. BEATs 6 lớp, FFN cắt 3072 → 1024 + mixup 0,5 (khoảng 29M tham số)
- `--ffn-dim 1024` giữ 1.024 nơ-ron FFN kích hoạt mạnh nhất (đo trên 16 batch train). Đang chạy.

| Epoch | Train acc (eval) | Val | mAP | Chênh lệch | Điểm |
|---|---|---|---|---|---|
| 1 | 0,352 | 0,308 | 0,331 | 0,044 | 0,286 |
| 2 | 0,627 | 0,437 | 0,450 | 0,190 | 0,342 |
| 3 | 0,803 | 0,463 | 0,489 | 0,339 | 0,294 |
| 4 | 0,883 | 0,484 | 0,517 | 0,399 | 0,284 |
| 5 | 0,941 | 0,508 | 0,539 | 0,433 | 0,292 |

| 6 | 0,969 | 0,497 | 0,531 | 0,472 | 0,260 |
| 7 | 0,975 | 0,494 | 0,537 | 0,481 | 0,253 |

- Kết thúc: dừng sớm ở epoch 7. Điểm (phạt 0,5) chọn epoch 2 (val 0,437, mAP 0,450, chênh lệch 0,19); val tốt nhất thật là 0,508 ở epoch 5. Hệ số phạt 0,5 quá mạnh, chọn checkpoint rất sớm, chưa hội tụ. Các run sau (E7 trở đi) dùng `--gap-penalty 0.2`.
- Kết luận: cắt FFN xuống 1024 giảm khoảng 18M tham số nhưng val đỉnh 0,508, thấp hơn E3 khoảng 0,07 và train đã 0,97. Cần chống overfit mạnh hơn mới có ích.

### E6. Như E3 với mixup 0,8 (dừng sớm do lỗi tiêu chí)
- `--mixup 0.8`, phạt gap 0,5, dừng sớm theo điểm. Tự dừng ở epoch 7 trong khi val vẫn đang tăng.

| Epoch | Train acc (eval) | Val | mAP | Chênh lệch |
|---|---|---|---|---|
| 1 | 0,483 | 0,399 | 0,432 | 0,084 |
| 2 | 0,747 | 0,506 | 0,546 | 0,241 |
| 3 | 0,865 | 0,538 | 0,579 | 0,327 |
| 4 | 0,929 | 0,539 | 0,589 | 0,389 |
| 5 | 0,953 | 0,548 | 0,607 | 0,405 |
| 6 | 0,972 | 0,545 | 0,602 | 0,426 |
| 7 | 0,977 | 0,561 | 0,611 | 0,416 |

- Bài học: phạt chênh lệch (hệ số 0,5) dùng cho quyết định dừng sớm làm run bị cắt khi val còn tăng. Đã sửa: dừng sớm theo val macro_acc, phạt chỉ dùng để chọn checkpoint (`best.pt` theo điểm, `best_val.pt` theo val).

### E7. Như E5 (6 lớp, FFN 1024, khoảng 29M tham số) với mixup 0,8 và phạt gap 0,2
- `--ffn-dim 1024 --mixup 0.8 --gap-penalty 0.2`, bản script cũ (dừng sớm theo điểm). **Bị mất khi sandbox hết hạn ở epoch 18/20.**

| Epoch | Train acc (eval) | Val | mAP | Chênh lệch |
|---|---|---|---|---|
| 1 | 0,357 | 0,302 | 0,328 | 0,055 |
| 2 | 0,608 | 0,425 | 0,454 | 0,182 |
| 3 | 0,775 | 0,461 | 0,496 | 0,314 |
| 4 | 0,868 | 0,484 | 0,526 | 0,384 |
| 5 | 0,927 | 0,497 | 0,530 | 0,430 |
| 6 | 0,946 | 0,509 | 0,556 | 0,436 |
| 7 | 0,963 | 0,504 | 0,544 | 0,458 |
| 8 | 0,972 | 0,516 | 0,562 | 0,456 |
| 9 | 0,986 | 0,524 | 0,569 | 0,462 |
| 10 | 0,989 | 0,521 | 0,567 | 0,469 |
| 11 | 0,993 | 0,527 | 0,569 | 0,466 |
| 12 | 0,992 | 0,530 | 0,575 | 0,463 |
| 13 | 0,995 | 0,531 | 0,577 | 0,464 |
| 14 | 0,994 | 0,527 | 0,576 | 0,467 |
| 15 | 0,998 | 0,534 | 0,579 | 0,464 |
| 16 | 0,998 | 0,536 | 0,582 | 0,462 |
| 17 | 0,998 | 0,535 | 0,580 | 0,463 |
| 18 | 0,997 | 0,534 | 0,582 | 0,463 |

- Kết luận: 29M tham số đạt val khoảng 0,535 và mAP khoảng 0,58, kém E3 (47,8M) khoảng 0,04 và 0,05. Chênh lệch train/val vẫn khoảng 0,46.

### E8. Như E6 (6 lớp, mixup 0,8) với dừng sớm theo val và phạt 0,2
- **Bị mất khi sandbox hết hạn ở epoch 11/20** (hạt giống cố định nên epoch 1 đến 7 trùng E6).

| Epoch | Train acc (eval) | Val | mAP | Chênh lệch |
|---|---|---|---|---|
| 8 | 0,988 | 0,551 | 0,610 | 0,437 |
| 9 | 0,993 | 0,562 | 0,615 | 0,430 |
| 10 | 0,992 | 0,564 | 0,6175 | 0,428 |
| 11 | 0,998 | 0,565 | 0,620 | 0,433 |

- Kết luận dở dang: val vẫn tăng chậm tới 0,565 và mAP 0,620 ở epoch 11, chưa biết có vượt 0,577 của E3 không.

## Sự cố
- Sandbox molab `sb-6dc3af564ac9e5e2` bị thu hồi (HTTP 410) khi E7 còn 2 epoch và E8 còn 9 epoch. Mất toàn bộ `/marimo` gồm checkpoint (`best.pt` của E3, E7, E8), cache n̂ đã giải nén, dataset đã tải, log chạy. Chỉ còn các số liệu ghi trong nhật ký này. Bản trên Drive còn nguyên: `nhat_cache.zip`, `best.pt` teacher, FRCRN. Checkpoint student chưa từng được sao lưu lên Drive.

## Cảnh báo về α của DeepNorm khi cắt lớp (phát hiện sau E8)
- `EncoderLayer.alpha = (2 * encoder_layers) ** 0.25` (src/model/beat/encoder.py). Teacher 12 lớp học với α = 24^0.25 = 2,21. `build_student` trong `scripts/run_beats_trunc.py` đặt `encoder_layers = L` nên student 6 lớp dùng α = 1,86 và student 4 lớp dùng α = 1,68, lệch so với trọng số đã học.
- Ảnh hưởng: E4 (4 lớp) lệch nhiều nhất nên kết luận "bỏ hai lớp mất khoảng 0,10 val" bị lẫn với tác động của α, chưa sạch. So sánh giữa các run 6 lớp (E3, E7, E8: mixup, FFN) vẫn công bằng vì cùng α.
- Cách sửa dự kiến: giữ α = 2,21 cố định khi cắt lớp (tách α khỏi `encoder_layers`), rồi chạy lại E3 và E4 để so lại. Chưa sửa mã, chưa chạy lại.

## Việc tiếp theo
1. Chốt công thức (mixup, KD, lr) theo điểm = val macro_acc − 0,5 × gap.
2. Giảm tham số: quét số lớp 4, 3, 2, 1 (mỗi lớp BEATs khoảng 7,1M, phần còn lại khoảng 5M); nếu cần nhỏ hơn thì thu hẹp chiều trong mô hình.
3. Cập nhật file này sau mỗi run.

## Loạt student ViT-S và student nhỏ (sandbox `sb-4dccc633b8d66e13`, teacher `best.pt` rồi fold0)

Student có bucket + embedding, bias vị trí và cổng GRU; 25 đến 30 epoch; chênh lệch train/val đo ở chế độ eval trên 2.000 clip train.

### ViT-S 21,49M với teacher `best.pt` (đến epoch 15), CSV `output/result/vits-kd-epoch15/`
| Run | Val tốt nhất | mAP | Chênh lệch |
|---|---|---|---|
| R0 (chỉ KD logit) | 0,513 (epoch 14) | 0,557 | 0,460 |
| R1 (+ q/k/v, MSE trên ma trận quan hệ, trọng số 0,1) | 0,508 (epoch 15) | 0,557 | 0,474 |
| R2 (+ q/k/v + bias vị trí) | 0,506 | 0,548 | 0,472 |
KD q/k/v và bias vị trí không đổi gì (khác nhau dưới 0,01); KD pos chỉ 0,0008 vì bảng bias khởi tạo bằng đúng nhóm head của teacher.

### Khối post-LN α trên ViT-S (CSV `vits-kd-alpha-postln/`)
Luồng A (post-LN α, bỏ LN1/LN2): 0,458; luồng B (post-LN α, giữ LN1/LN2): 0,512, ngang R0. Bỏ LN1/LN2 làm kém hơn 0,05 đến 0,07.

### Teacher fold0 (SHA-256 `60a037f5111c178e...`), chống overfit: lr 1e-4 tự giảm, dropout 0,2, weight decay 0,1 (CSV `vits-kd-fold0/`)
Teacher fold0 trên n̂ cache: top-1 0,6636, mAP 0,7356; theo SNR −5 dB 0,741, 0 dB 0,731, 5 dB 0,696, 10 dB 0,669, 15 dB 0,620, 20 dB 0,524.
| Run | Val tốt nhất | mAP | Chênh lệch |
|---|---|---|---|
| R0 pre-LN không α | 0,483 (epoch 21) | 0,527 | 0,419 |
| B post-LN α, có LN1/LN2, 25 epoch | 0,480 (epoch 13) | 0,520 | 0,394 |
B không hơn R0 (khác nhau dưới 0,015); chạy hai lần cho kết quả sát nhau.

### Student nhỏ (kiểu B, trừ CED-Tiny), số liệu từng epoch trong `imp/kd-v4/results/tiny_students_from_monitor.csv`
| Run | Tham số | Khởi tạo | Val tốt nhất | mAP | Chênh lệch | Ghi chú |
|---|---|---|---|---|---|---|
| CED-Tiny, pre-LN, không dropout | 5,43M | AudioSet | **0,629** (epoch 12) | 0,701 | 0,223 | chạy dở ở epoch 16/30 khi sandbox hết hạn |
| DeiT-Ti, post-LN α | 5,45M | ImageNet | 0,463 | 0,501 | 0,351 | dừng ở khoảng epoch 24/30 |
| ViT-Tiny, post-LN α | 5,45M | ImageNet-21k | 0,455 | 0,489 | 0,297 | dừng sớm ở epoch 22 |

### CED-Tiny: chẩn đoán
- CED-Tiny cấu trúc B (post-LN α) sụp ngay: val 0,0278 (bằng ngẫu nhiên). Cấu trúc gốc pre-LN cũng sụp ở epoch 1 đến 3 (val 0,032 đến 0,038) cả khi bảng bias khởi tạo bằng 0.
- Linear probe trên đặc trưng đóng băng (3.000 clip): DeiT ImageNet 0,270 (CLS) và 0,263 (mean pooling); CED-Tiny 0,489 (CLS) và 0,510 (mean pooling): đặc trưng CED tốt, đầu vào đúng.
- Thống kê kích hoạt lúc khởi tạo: CED patch embed std 12,5 và embedding vị trí std 2,6; DeiT ngẫu nhiên 0,55 và 0,02. Kích hoạt lớn làm cấu trúc post-LN α khuếch đại sublayer.
- Bốn thí nghiệm ngắn (4.000 clip, val sau epoch 1 và 2): sạch 0,052 và 0,135; mixup 0,8 0,048 và 0,127; dropout 0,2 0,035 và 0,042; dropout, lr 2e-5 và mixup 0,032 và 0,036. **Thủ phạm là dropout**, không phải mixup hay lr.
- Chạy lại với dropout 0, cấu trúc pre-LN gốc, bias bằng 0: epoch 1 val 0,568, epoch 2 0,605, epoch 3 0,618, đỉnh 0,629 ở epoch 12, mAP 0,70, chênh lệch train/val chỉ 0,22.

### Sự cố thứ hai
Sandbox `sb-4dccc633b8d66e13` bị thu hồi (HTTP 410) khi đang chạy CED-Tiny, DeiT-Ti và đánh giá theo SNR. Mất checkpoint và lịch sử đầy đủ; xem `imp/kd-v4/README.md` mục "Đã mất".
