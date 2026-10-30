# kd-v5: BEATs → CED-Tiny, dự án chưng cất tự chứa

Dự án độc lập trong `imp/kd-v5`, chỉ gồm mã Python, YAML, test và tài liệu. Điểm vào duy nhất là `main.py`. Mục tiêu là tái lập công thức CED-Tiny tốt nhất của `imp/kd-v4` (val macro_acc 0,629; mAP 0,701) với teacher BEATs fold0, đồng thời bổ sung cơ chế nối tiếp lần chạy và sao lưu định kỳ.

## Công thức

| Thành phần | Giá trị |
|---|---|
| Teacher | BEATs fold0 (`beats_teacher_fold0_weights.pt`), đóng băng |
| Student | DeiT-Tiny + trọng số CED-Tiny AudioSet (`mispeech/ced-tiny`), 5,42M tham số |
| Đầu vào | n̂ lấy từ cache, chuẩn hoá RMS 0,07925 |
| Log-mel | 64 mel, 398 khung hình, crop 4 giây |
| Regularization | dropout 0, attention dropout 0, drop-path 0, mixup 0,8 (Beta 0,4) |
| Loss | CE label smoothing 0,1 + logit KD (T=4, weight=1,0) |
| Optimizer | AdamW, lr 1e-4, weight decay 0,1, warmup 500 bước |
| Chọn epoch | `score = macro_acc - 0,2 × gap(train_eval, val)`; early stop sau 7 epoch không cải thiện |
| Không dùng | KD q/k/v hoặc bias vị trí (kd-v4 cho thấy không có lợi) |

## Cần có để chạy thật

- Dataset `mixture-36l` (mặc định `D:\content\mix-dataset`, đọc qua `config/url.yaml`).
- Cache n̂ đã tính trước, mỗi split một cặp `train.npy` / `train.done`, `validation.npy` / `validation.done` (và `test.npy` / `test.done` nếu đánh giá test). `.done` chứa đúng số mẫu của manifest.
- Teacher `pretrained/beat-ft/beats_teacher_fold0_weights.pt` (đường dẫn tương đối trong `config/kd_ced_tiny.yaml`).
- Trọng số CED-Tiny: tải từ Hugging Face, hoặc trỏ `--ced-weights` / `student.ced_path` tới `model.safetensors` **đúng CED-Tiny** (embedding 192 chiều). File `../../pretrained/model.safetensors` của repo là model 9,7M / 256 chiều, không phải CED-Tiny và sẽ bị từ chối.

## Lệnh

```powershell
# Xem bảng tham số (không tải trọng số)
python main.py params

# Smoke test: dữ liệu tổng hợp, teacher BEATs khởi tạo ngẫu nhiên, 1 epoch
python main.py smoke

# Huấn luyện thật (tự tải CED-Tiny từ Hugging Face)
python main.py train --noise-cache <thư-mục-cache>

# Chạy offline: trỏ tới model.safetensors đúng CED-Tiny (192 chiều)
python main.py train --noise-cache <cache> --ced-weights <ced-tiny-model.safetensors>

# Nối tiếp đúng lần chạy cũ (không tạo thư mục mới)
python main.py train --noise-cache <cache> `
  --run-id kd-ced-tiny-2026-10-27-10-00-00

# Đánh giá checkpoint trên test (không đụng test trong lúc train)
python main.py eval --checkpoint output/checkpoint/<run-id>/student/last_best_checkpoint.pt `
  --noise-cache <cache> --split test
```

Mặc định `train` chỉ đánh giá validation. `eval --split test` mới đọc manifest test; có thể truyền `--sample-ids-file` để dùng danh sách đã lọc rò rỉ (`test_clean.csv`, `validation_clean.csv`).

## Đầu ra

Theo `src.utils.paths.RunPaths`:

```
output/log/<run-id>.log
output/checkpoint/<run-id>/student/last_best_checkpoint.pt
output/checkpoint/<run-id>/student/train_state.pt
output/result/<run-id>/result.csv
output/result/<run-id>/history/student.json
```

- `last_best_checkpoint.pt`: trọng số của epoch có `score` tốt nhất, kèm `epoch` và `score`.
- `train_state.pt`: trọng số cuối epoch, optimizer, scheduler, `best_score`, `stale`, `lr_scale`, lịch sử. Đây là thứ `--run-id` dùng để nối tiếp.
- `result.csv`: một dòng tổng thể và một dòng cho mỗi mức SNR (−5, 0, 5, 10, 15, 20), gồm 9 chỉ số, `macro_acc`, `mAP_macro`, `score`, `gap`.
- `--backup-dir <thư-mục>` sao chép checkpoint, result và history sau mỗi epoch (dùng cho ổ Drive được mount). Không chứa token.

## Khác biệt so với kd-v4

- Có resume thật: `train_state.pt` lưu cả optimizer/scheduler và lịch sử; `--run-id` chạy tiếp từ epoch kế tiếp.
- Có `eval` riêng cho validation/test và xuất CSV theo SNR; train không tự động chạm test.
- Chỉ giữ logit KD, CE, mixup, CED-Tiny; bỏ q/k/v, bias, ReviewKD để dự án gọn và đúng với cấu hình đã thắng.
- Không notebook, không phụ thuộc `scripts/run_vits_kd.py` hay mã ngoài thư mục này.
- Có test tổng hợp cho student, loss, metrics, dataset cache, trainer và resume.

## Kết quả tham chiếu

kd-v4 đạt **val macro_acc 0,629 / mAP 0,701** ở epoch 12 với CED-Tiny, nhưng checkpoint đã mất khi sandbox hết hạn. kd-v5 chưa được huấn luyện thật trong phiên này; con số trên là mục tiêu tái lập, không phải kết quả mới. Sau khi chạy, ghi kết quả thật vào `results/` hoặc README này, không thay bằng số ước lượng.

## Lưu ý

- Cache n̂ phải khớp **đúng thứ tự** manifest; `NoiseCacheDataset` kiểm tra số dòng và file `.done`, không tự sắp xếp lại.
- `student.ced_path`/`--ced-weights` chỉ dùng với `model.safetensors` của CED-Tiny (embedding 192 chiều, khớp DeiT-Tiny); nếu bỏ trống sẽ tải từ Hugging Face. `load_ced_tiny` báo lỗi rõ nếu file sai chiều.
- CED-Tiny rất nhạy với dropout: giữ dropout 0 như cấu hình. Nếu đổi sang post-LN, kết quả có thể sụp (xem kd-v4).
- Val/test gốc còn khoảng 50% clip có noise ytid trùng AudioSet; dùng `validation_clean.csv` / `test_clean.csv` nếu cần báo cáo sạch rò rỉ. Việc chọn epoch vẫn nên dựa trên validation đầy đủ để so sánh với kd-v4.
