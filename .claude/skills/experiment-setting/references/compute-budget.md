# Quản lý tài nguyên tính toán

Cấu hình phần cứng tham chiếu theo người dùng: thường là một GPU **RTX 6000 với 96 GB VRAM**. Mọi con số dưới đây là ước lượng; **đo trước khi tin**.

## Mục lục
1. Bộ nhớ cho tham số và trạng thái tối ưu
2. Bộ nhớ cho activation
3. Ước lượng thời gian
4. Quy trình đo
5. Bảng tham chiếu của repo
6. Khi hết bộ nhớ hoặc quá chậm

## 1. Bộ nhớ cho tham số và trạng thái tối ưu

Với AdamW, trọng số fp32 và autocast bf16:

| Thành phần | Byte mỗi tham số |
|---|---:|
| Trọng số fp32 | 4 |
| Gradient fp32 | 4 |
| Moment bậc 1 và 2 của Adam | 8 |
| **Cộng** | **16** |
| EMA (nếu có) | +4 |

Ví dụ:

| Mô hình | Tham số | Trạng thái huấn luyện (16 B) | Có EMA (20 B) |
|---|---:|---:|---:|
| KD-MoE S (≈1,25M) | 1,25M | ≈ 20 MB | ≈ 25 MB |
| KD-MoE M (≈6,45M) | 6,45M | ≈ 103 MB | ≈ 129 MB |
| KD-MoE L (≈14,2M) | 14,2M | ≈ 227 MB | ≈ 284 MB |
| BEATs base (ước, ≈ 90M; chưa đếm trong repo) | ≈ 90M | ≈ 1,4 GB | ≈ 1,8 GB |

Kết luận: với student của repo, **tham số không phải nút thắt**; activation và dữ liệu mới là.

## 2. Bộ nhớ cho activation

Không có công thức gọn đáng tin cho mọi kiến trúc; ước lượng cấp bậc:

- Activation tỉ lệ với `batch × số token × d × số lớp × hệ số`; student có 192 token và d ≤ 384.
- Attention có thêm `batch × heads × tokens²` cho ma trận điểm (192² nhỏ; BEATs trên 4 s cũng ở mức này).
- bf16 giảm phần activation còn khoảng một nửa so với fp32.
- Gradient checkpointing đổi thêm tính toán lấy bộ nhớ khi cần.

`doc/moe.md` §8.5 ước lượng thô (chưa đo): cấu hình L, batch 64, dưới 10 GB đỉnh ở bf16. Với 96 GB, batch 128 đến 256 khả thi theo ước lượng đó, nhưng **đổi batch phải đổi lr và so sánh lại**.

## 3. Ước lượng thời gian

```
steps_per_epoch = ceil(N_train / batch)
thời gian ≈ epochs × steps_per_epoch × giây_mỗi_bước + đánh giá + ghi checkpoint
```

Ví dụ minh hoạ (số mẫu từ log chạy thật): train 30 240 mix, batch 64 → 473 bước mỗi epoch. Nhân với `giây_mỗi_bước` **đo được**, không dùng số từ máy khác.

Thành phần dễ bị quên:

| Thành phần | Ghi chú |
|---|---|
| Đọc dữ liệu | Mix wav 4 s đọc từ đĩa; kiểm tra GPU có bị đói dữ liệu không (`num_workers`) |
| Log-mel trên GPU hay CPU | CPU có thể là nút cổ chai |
| Đánh giá mỗi epoch | Val 6 480 mix; chạy ở `eval_batch_size` lớn hơn |
| Teacher cache | Chạy một lần; student đọc lại, không tính lại |
| Dừng sớm | `patience` 15 nghĩa là tối đa `best + 15` epoch; ngân sách theo `epochs` tối đa |
| Nhiều seed | Thời gian × số seed; lập kế hoạch ngay từ đầu |

Kế hoạch tổng một ablation: `số cấu hình × số seed × thời gian một lần chạy`. Ví dụ A0 + 3 ablation × 3 seed = 12 lần chạy; nếu mỗi lần X giờ thì tổng 12X.

## 4. Quy trình đo

1. Chạy 50 đến 100 bước huấn luyện thật (không smoke 1 epoch), bỏ 10 bước đầu (khởi động, cuDNN).
2. Đồng bộ trước khi bấm giờ: `torch.cuda.synchronize()`.
3. Đọc đỉnh bộ nhớ: `torch.cuda.max_memory_allocated()` và `max_memory_reserved()`.
4. Tính `giây_mỗi_bước` và `mẫu_mỗi_giây`, nhân ra epoch và tổng.
5. Ghi kết quả đo vào log chạy để lần sau khỏi đoán.

```python
torch.cuda.reset_peak_memory_stats()
for step, batch in zip(range(60), loader):
    if step == 10:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    ...  # forward, backward, step
torch.cuda.synchronize()
sec_per_step = (time.perf_counter() - t0) / 50
peak_gb = torch.cuda.max_memory_allocated() / 2**30
```

Chọn batch lớn nhất vừa bộ nhớ bằng `src/utils/batch_size.py::largest_fitting_batch` (`vram_fraction` mặc định 0,9).

## 5. Bảng tham chiếu của repo

| Mục | Giá trị | Nguồn |
|---|---|---|
| Student S, M, L | ≈ 1,25M / 6,45M / 14,2M tham số | `config/kdmoe_*.yaml`, `doc/moe-molab-run.md` |
| Giai đoạn 1 | 8 epoch | `train.stage1_epochs` |
| Giai đoạn 2 | tối đa 120 epoch, patience 15 | `config/kdmoe_m.yaml` |
| Batch | 64 | `train.batch_size` |
| Sandbox molab | giới hạn 12 giờ, có resume | `doc/moe-molab-run.md` §8 |
| Teacher | chạy một lần, cache `teacher_{split}.npz` | `scripts/teacher_cache.py` |

## 6. Khi hết bộ nhớ hoặc quá chậm

| Triệu chứng | Thử |
|---|---|
| OOM | Giảm batch (rồi dùng accumulation), bf16, gradient checkpointing, giảm `num_frames` |
| GPU dưới 50% sử dụng | Tăng `num_workers`, `pin_memory`, đọc dữ liệu vào RAM, chuyển log-mel lên GPU |
| Epoch chậm do đánh giá | Đánh giá thưa hơn hoặc trên tập con khi dò, đầy đủ khi chốt |
| Hết thời gian sandbox | Dựa vào `resume.pt` và đồng bộ Drive mỗi epoch |
| Nhiều cấu hình | Chạy khảo sát 1 seed trước, chỉ chạy 3 seed cho cấu hình thắng |
