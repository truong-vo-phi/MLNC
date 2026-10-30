# Debug mô hình

## Mục lục
1. Thứ tự kiểm tra
2. Overfit một batch nhỏ
3. Kiểm tra shape và gradient
4. Phát hiện rò rỉ dữ liệu
5. Đọc đường cong loss
6. NaN và bất ổn số

## 1. Thứ tự kiểm tra

Đi từ rẻ đến đắt, dừng khi tìm ra:

1. **Dữ liệu:** nghe hoặc vẽ vài mẫu; nhãn có khớp; độ dài và biên độ đúng.
2. **Shape và dtype** đi qua từng khối.
3. **Overfit một batch nhỏ.** Không overfit được thì lỗi nằm ở mã, không phải ở dữ liệu.
4. **Gradient:** có chảy, có đúng chỗ, độ lớn hợp lý.
5. **Loss và đường cong** trên tập lớn.
6. **Rò rỉ và đánh giá.**

## 2. Overfit một batch nhỏ

```python
batch = next(iter(loader))           # 8 đến 32 mẫu
model.train()
for step in range(300):
    out = model(batch["waveform"])
    loss = criterion(out, batch)["ce"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 50 == 0:
        print(step, loss.item(), (out.logits.argmax(-1) == batch["label"]).float().mean().item())
```

Kỳ vọng: loss về gần 0, accuracy về 100%. Chú ý:

- Tắt ngẫu nhiên khi kiểm: `dropout = 0`, `drop_path_max = 0`, tắt SpecAugment (nó chỉ chạy khi `training`), `label_smoothing = 0` nếu muốn loss về sát 0.
- Dùng cùng một batch cố định, không lấy batch mới mỗi bước.
- Không overfit được thì nghĩ đến: nhãn lệch hàng, `lr` quá nhỏ/lớn, tham số bị `requires_grad=False`, `optimizer` không chứa tham số, `zero_grad` đặt sai chỗ.
- Với KD-MoE: kiểm cả `ce` và `kd`; router phải khớp `p_t` trên batch nhỏ.

## 3. Kiểm tra shape và gradient

**Shape:** in shape đầu ra từng khối với batch nhỏ, so với bảng trong `doc/moe.md` (ví dụ `(B, 8, 24, d)` sau PatchEmbed). Thêm test cố định shape trong `tests/test_*_model.py`.

**Gradient:**

```python
loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad:
        g = p.grad
        print(name, None if g is None else f"{g.norm().item():.3e}")
```

Diễn giải:

| Quan sát | Ý nghĩa |
|---|---|
| `grad is None` ở tham số cần học | Khối không nằm trên đường tính loss, hoặc bị `detach` nhầm |
| Norm bằng 0 | Hoạt hoá chết, cổng đóng hết, hoặc loss không phụ thuộc |
| Norm rất lớn (≫ `grad_clip`) | Bùng nổ; xem lr, chuẩn hoá |
| Norm lệch hàng chục lần giữa các khối | Khởi tạo hoặc tỉ lệ loss mất cân đối |

Kiểm riêng cho giai đoạn đóng băng: ở giai đoạn 1 của KD-MoE các khối `vit.`, `final_ln.`, `pool_gru.`, `router.`, `experts.` phải có `requires_grad=False`.

Kiểm số học bằng phương pháp sai phân hữu hạn cho một loss mới: so `torch.autograd.gradcheck` (dùng `float64`, mô hình nhỏ).

## 4. Phát hiện rò rỉ dữ liệu

Rò rỉ làm số liệu val tốt giả. Các nguồn có thật trong bài toán này:

| Nguồn | Cách kiểm |
|---|---|
| Hai mix cùng clip nhiễu rơi vào train và val | Chia theo `noise_ytid` hoặc `noise_clip_id`; kiểm giao tập id giữa split bằng rỗng |
| Cùng tiếng nói (utterance) ở nhiều split | Kiểm giao tập `speech_id`/`clean_path` giữa split |
| Probe của teacher dự đoán mẫu nó đã huấn luyện | Dùng cross-fit và báo số out-of-fold |
| Teacher đã thấy nhãn (BEATs fine-tune AudioSet) | Dùng bản pretrain thuần, ghi rõ nguồn checkpoint |
| Chuẩn hoá dùng thống kê của val/test | Tính mean/std chỉ trên train |
| `p_t` cache khớp sai `mix_id` | Kiểm khoá cache, tổng `p_t` theo nhóm = 1 |

```python
train_ids = set(train_manifest["noise_clip_id"])
val_ids = set(val_manifest["noise_clip_id"])
assert not (train_ids & val_ids), "rò clip nhiễu giữa train và val"
```

Dấu hiệu nghi ngờ: val gần bằng hoặc cao hơn train từ đầu; số liệu tốt bất thường khi đổi từ một cấu hình nhỏ lên lớn không cải thiện thêm.

## 5. Đọc đường cong loss

| Hình dạng | Nguyên nhân thường gặp | Việc thử |
|---|---|---|
| Phẳng từ đầu, ≈ ln K (ln 36 ≈ 3,58) | Mô hình không học: lr, nhãn, `zero_grad`, dữ liệu | Overfit một batch |
| Giảm rồi nổ NaN | lr quá lớn, fp16, log của 0 | Giảm lr, bf16, `eps`, grad clip |
| Train giảm, val phẳng hoặc tăng | Overfit | Regularization, dữ liệu, dung lượng nhỏ hơn |
| Train và val đều cao, chậm | Underfit hoặc lr thấp | Tăng dung lượng, tăng lr, huấn luyện lâu hơn |
| Răng cưa mạnh | Batch nhỏ, lr cao, dữ liệu không xáo | Tăng batch, giảm lr, EMA |
| Val tốt hơn train | Dropout, drop-path, SpecAugment chỉ chạy lúc train | Bình thường |
| Loss tổng giảm nhưng một thành phần tăng | Trọng số loss mất cân đối | Ghi và vẽ từng thành phần (`ce`, `kd`, `front`, `cons`) |

Luôn ghi từng thành phần loss mỗi epoch (trainer đã ghi `ce`, `kd`, `front`, `cons`, `gate_mean`, `gate_low`).

Giai đoạn 2 KD-MoE: theo dõi thêm `router_acc` theo SNR và KL(p_t ‖ p_s) trên val.

## 6. NaN và bất ổn số

1. `torch.autograd.set_detect_anomaly(True)` để tìm phép sinh NaN (chậm, chỉ để debug).
2. Kiểm đầu vào: `torch.isfinite(batch["waveform"]).all()`; clip nhiễu rỗng làm công suất 0.
3. `log(x)` với x ≥ 0: thêm `eps` (`1e-6` trong log-mel).
4. Softmax/KL ở bf16: ép `float()` (repo đã làm trong loss).
5. Chia cho `std` rất nhỏ khi chuẩn hoá theo clip: `std + 1e-5`.
6. Tắt AMP để xem NaN có biến mất không; nếu có thì tìm phép bf16 gây ra.
