# kd-v4: student nhỏ chưng cất từ BEATs (fold0), phát hiện CED-Tiny, snapshot ngày 2026-10-26

Snapshot mã, cấu hình và kết quả của loạt thí nghiệm student ViT-S, ViT-Tiny, DeiT-Ti, CED-Tiny có bucket + embedding, bias vị trí và cổng GRU. **Không gồm dữ liệu, checkpoint, cache n̂** (xem mục "Đã mất").

## Kết luận chính

| Student | Tham số | Khởi tạo | Val macro_acc tốt nhất | mAP | Chênh lệch train/val |
|---|---|---|---|---|---|
| **CED-Tiny**, pre-LN gốc, không dropout | 5,43M | AudioSet (mispeech/ced-tiny) | **0,629** (epoch 12) | 0,701 | 0,223 |
| ViT-S, pre-LN (R0) | 21,49M | ImageNet | 0,483 | 0,527 | 0,419 |
| ViT-S, post-LN α, có LN1/LN2 (B) | 21,49M | ImageNet | 0,480 | 0,520 | 0,394 |
| DeiT-Ti kiểu B | 5,45M | ImageNet | 0,463 | 0,501 | 0,351 |
| ViT-Tiny kiểu B | 5,45M | ImageNet-21k | 0,455 | 0,489 | 0,297 |
| Teacher BEATs fold0 (tham chiếu) | khoảng 90M | AudioSet | 0,664 | 0,736 | |

Teacher fold0 theo SNR (độ chính xác): −5 dB 0,741; 0 dB 0,731; 5 dB 0,696; 10 dB 0,669; 15 dB 0,620; 20 dB 0,524.

- **Nguồn khởi tạo là yếu tố lớn nhất:** CED-Tiny (AudioSet) vượt mọi student ImageNet khoảng 0,15 và đạt mục tiêu 0,6 đến 0,7 chỉ với 5,43M tham số. Đặc trưng đóng băng đã cho thấy khoảng cách này: linear probe trên 3.000 clip, DeiT ImageNet 0,27 (CLS) so với CED 0,49 (CLS), 0,51 (mean pooling).
- **KD q/k/v và bias vị trí không giúp** (R0, R1, R2 với ViT-S: khác nhau dưới 0,01 val), và post-LN α với LN1/LN2 không giúp so với pre-LN.
- **Bỏ LayerNorm 1 và 2 gây hại** (val thấp hơn 0,05 đến 0,07).

## CED-Tiny: những điều phải biết để tái hiện
- CED-Tiny **sụp về mức ngẫu nhiên khi có dropout 0,2** (kể cả attention dropout và drop-path): thí nghiệm chẩn đoán trên 4.000 clip, val sau 2 epoch: sạch 0,135; mixup 0,127; dropout 0,042; dropout + lr thấp + mixup 0,036. Dùng dropout 0 và giữ mixup 0,8, weight decay 0,1.
- Cấu trúc **post-LN α không dùng được với CED**: kích hoạt của CED rất lớn (patch embed std 12,5, embedding vị trí std 2,6 so với 0,55 và 0,02 của DeiT ngẫu nhiên), LN(α x + sublayer) chuẩn hoá lại dòng residual làm sublayer bị khuếch đại. Dùng cấu trúc gốc pre-LN (`--ced-tiny --n-mels 64 --bias-init zero`, không `--post-ln`).
- Lệnh đã dùng: `python -m scripts.run_vits_kd --noise-cache <cache> --name t-ced-tiny-pre1 --epochs 30 --arch deit_tiny_patch16_224 --ced-tiny --n-mels 64 --bias-init zero --lr 1e-4 --lr-patience 1 --dropout 0 --attn-dropout 0 --drop-path 0 --weight-decay 0.1 --patience 7`.

## Nội dung

| Đường dẫn | Là gì |
|---|---|
| `src/`, `scripts/`, `config/` | Mã nguồn hiện tại. Mới: `src/model/baseline/vit_bias.py` (ViT/DeiT/CED có bias và cổng, `AlphaBlock`, `load_ced_tiny`), `src/loss/beat_attention_kd.py` (KD q/k/v và bias), `scripts/run_vits_kd.py` (huấn luyện), `scripts/eval_vits_snr.py` (đánh giá theo SNR), `scripts/eval_teacher_cache.py`, `scripts/check_kd_init.py`, `scripts/run_beats_trunc.py` (student BEATs cắt lớp) |
| `config/reviewkd_deit.yaml` | Cấu hình dùng chung; teacher trỏ tới `pretrained/beat-ft/beats_teacher_fold0_weights.pt` |
| `results/experiments.md` | Nhật ký từng lần thử (E2 đến E8, ViT-S, student nhỏ) |
| `results/vits-kd-epoch15`, `vits-kd-alpha-postln`, `vits-kd-fold0` | CSV đầy đủ từng epoch của R0, R1, R2, hai luồng α, B và R0 với teacher fold0 |
| `results/tiny_students_from_monitor.csv` | Số liệu từng epoch của CED-Tiny, DeiT-Ti, ViT-Tiny **ghi lại từ thông báo giám sát** (thiếu một vài epoch, xem dưới) |
| `molab_helpers/`, `lock.txt`, `pyproject.toml` | Như kd-v3 |

## Đã mất (sandbox hết hạn, HTTP 410) và chưa sao lưu
- Toàn bộ checkpoint student (`best.pt`, `best_val.pt`), kể cả CED-Tiny.
- `history.json` và log đầy đủ của CED-Tiny, DeiT-Ti, ViT-Tiny; file CSV trong `tiny_students_from_monitor.csv` chỉ còn các epoch đã được giám sát báo (CED-Tiny thiếu epoch 15, đang chạy dở ở epoch 16 trên 30; DeiT-Ti thiếu epoch 23 và dừng ở khoảng epoch 24 trên 30; ViT-Tiny dừng sớm ở epoch 22).
- Đánh giá theo từng mức SNR của các student (script đã viết, chưa kịp chạy xong).
- Bốn thí nghiệm chẩn đoán dropout và kết quả linear probe chỉ còn dạng số ghi trong README này.

## Cần có ngoài gói này để chạy lại
Dataset `mixture-36l` (Kaggle, cần `KAGGLE_ACCESS_TOKEN`), cache n̂ `FA26_Capstone/Truong/beat-mv-kd-v3/nhat/nhat_cache.zip` (Drive), teacher `beats_teacher_fold0_weights.pt` (SHA-256 bắt đầu `60a037f5111c178e`), trọng số CED-Tiny tự tải từ Hugging Face `mispeech/ced-tiny`.

## Lưu ý vận hành đã rút ra
- Sandbox molab bị thu hồi sau vài giờ; **sao lưu checkpoint và `history.json` lên Drive sau mỗi epoch**, chưa làm trong `run_vits_kd.py`.
- `run_vits_kd.py` chưa có cơ chế nối tiếp (resume).
- Với `execute-code.sh`, mã chạy lâu phải đặt trong tiến trình nền; lời gọi chạy quá khoảng một phút sẽ bị ngắt.
