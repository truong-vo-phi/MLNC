# Mẫu PyTorch dùng trong repo

## Mục lục
1. Dataset và DataLoader
2. Custom loss
3. Mixed precision
4. Gradient accumulation
5. Huấn luyện phân tán cơ bản (DDP)
6. Tránh các lỗi phổ biến

## 1. Dataset và DataLoader

Họ lớp: `AudioDataset` (`dataset_key`, `__len__`, `__getitem__`), trả về `dict` với khoá cố định, ví dụ `waveform` (mixture), `noise`, `snr_db`, `label`, `group`, `noise_ytid`, `teacher_logits36`.

Nguyên tắc:

- `__getitem__` trả tensor đã đúng dtype và độ dài (4 s = 64 000 mẫu); đệm hoặc cắt trong dataset, không trong vòng huấn luyện.
- Mix cố định, không sinh online: `p_t` cache theo `mix_id`, biến đổi online làm cache sai.
- Batch cần cấu trúc cặp (cùng clip nhiễu cho `L_cons`) thì dùng sampler (`PairedClipSampler`), không đoán trong collate.
- `num_workers > 0` kèm `persistent_workers=True` khi epoch ngắn; `pin_memory=True` khi chuyển GPU.

```python
loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    sampler=sampler,            # hoặc shuffle=True nếu không có sampler
    num_workers=4,
    pin_memory=True,
    drop_last=True,             # tránh batch cuối quá nhỏ làm lệch cặp và thống kê
    persistent_workers=True,
)
```

## 2. Custom loss

Kế thừa lớp phù hợp; `name` duy nhất; `forward` nhận `(prediction, target)` hoặc `(student, teacher)`; đăng ký trong `src/loss/__init__.py`.

```python
class FrontLoss(SupervisedLoss):
    name = "front"

    def __init__(self, lam_over: float = 2.0) -> None:
        super().__init__()
        self.lam_over = lam_over

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction.float() - target.float()      # fp32 cho loss
        weight = 1.0 + (self.lam_over - 1.0) * (diff > 0).float()
        return (weight * diff.abs()).mean()
```

Việc cần làm mỗi khi viết loss:

- Ép `float()` cho logit trước log, softmax, KL (bf16 làm tròn thô).
- `detach()` mục tiêu đến từ mô hình khác (teacher).
- Khai báo phép giảm (`mean`) rõ ràng; ghi lại chuẩn hoá theo batch hay theo token.
- Thêm test nhỏ trong `tests/test_*_losses.py` với giá trị tính tay.

## 3. Mixed precision

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(batch["waveform"])
loss = criterion(output, batch)        # criterion ép float() bên trong
loss.backward()
```

- **bf16:** cùng dải số mũ với fp32 nên không cần `GradScaler`. Đây là lựa chọn của repo (`amp: bf16`).
- **fp16:** cần `torch.amp.GradScaler`, dễ NaN khi log nhiễu gần 0.
- Giữ ở fp32: loss, softmax, KL, log-mel (`log(x + 1e-6)`).
- Trọng số vẫn fp32; autocast chỉ đổi dtype phép tính.

## 4. Gradient accumulation

Repo **chưa có**. Mẫu đúng:

```python
accum = 4                                   # batch hiệu dụng = batch_size * accum
optimizer.zero_grad(set_to_none=True)
for step, batch in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(batch["waveform"]), batch)["total"] / accum
    loss.backward()
    if (step + 1) % accum == 0:
        clip_grad_norm_(params, config.grad_clip)
        optimizer.step()
        scheduler.step()                    # theo bước tối ưu, không theo bước micro-batch
        optimizer.zero_grad(set_to_none=True)
```

Cảnh báo riêng của repo:

- **`L_cons` cần cặp trong cùng micro-batch.** Cặp được tạo từ hai mix cùng clip nhiễu; nếu hai mix rơi vào hai micro-batch khác nhau thì loss mất cặp. Giữ nguyên cấu trúc "N clip nhiễu × 2 mix" **bên trong mỗi micro-batch**.
- EMA cập nhật theo bước tối ưu, không theo micro-batch.
- Số bước warmup và tổng bước tính theo bước tối ưu.
- Đổi batch hiệu dụng thì đổi lr và so sánh lại (`doc/moe.md` §8.5).

## 5. Huấn luyện phân tán cơ bản (DDP)

Repo **chưa có**. Khi thật sự cần nhiều GPU:

```python
# chạy: torchrun --nproc_per_node=N train.py
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
set_seed(config.seed + rank)                # khác seed theo rank cho augment ngẫu nhiên

model = DDP(model.to(rank), device_ids=[rank])
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=per_gpu_batch, sampler=sampler, ...)

for epoch in range(epochs):
    sampler.set_epoch(epoch)                # bắt buộc, nếu không thứ tự lặp lại mỗi epoch
    ...

if rank == 0:
    torch.save(model.module.state_dict(), path)   # lưu .module, tránh tiền tố "module."
```

Lưu ý:

- Batch hiệu dụng = `per_gpu_batch × world_size × accum`; scale lr cho phù hợp.
- Chỉ rank 0 ghi log, checkpoint, đánh giá; các rank khác chờ `dist.barrier()`.
- Tham số không dùng trong forward cần `find_unused_parameters=True` (chậm hơn); giai đoạn 1 của KD-MoE đóng băng nhiều khối nên để ý.
- Với student 6,4M tham số trên một GPU lớn, DDP thường **không cần**: dữ liệu và nhỏ mô hình khiến tăng batch trên một GPU rẻ hơn.

## 6. Các lỗi phổ biến

| Lỗi | Dấu hiệu | Sửa |
|---|---|---|
| Quên `model.eval()` / `torch.no_grad()` khi đánh giá | Val dao động, hết bộ nhớ | Dùng `evaluate` của `Trainer`, nó bọc sẵn |
| Quên `detach` teacher | KD gradient chảy vào teacher | `teacher_logits.detach()` |
| Trộn thứ tự chiều | Shape đúng nhưng kết quả vô nghĩa | In shape sau mỗi khối; test shape trong `tests/` |
| `drop_last=False` với cặp | Batch cuối lẻ làm sampler vỡ cặp | `drop_last=True` |
| Tên khoá yaml sai | Giá trị mặc định âm thầm được dùng | `BaseConfig.from_dict` báo khoá lạ; chạy `scripts/describe_config.py` |
