---
name: implement-setting
description: Cài đặt, huấn luyện và gỡ lỗi mô hình PyTorch trong repo MLNC theo chuẩn của dự án. Bao gồm viết Dataset/DataLoader và sampler, custom loss, mixed precision (bf16), gradient accumulation, huấn luyện phân tán cơ bản; giữ cấu hình tách khỏi mã (YAML + BaseConfig), cố định seed, môi trường và Git để chạy lại được; debug mô hình (overfit một batch nhỏ, kiểm tra shape và gradient, phát hiện rò rỉ dữ liệu, đọc đường cong loss); và dùng hoặc sửa mã có sẵn (repo bài báo, fairseq, HuggingFace, nạp checkpoint pretrain) mà không làm hỏng. Hãy dùng skill này mỗi khi người dùng viết hoặc sửa model, loss, trainer, dataset, nạp trọng số, gặp NaN, loss không giảm, kết quả không tái lập, hết bộ nhớ GPU, hoặc cần chạy thí nghiệm lâu có resume, kể cả khi họ không nhắc tên skill. Use for PyTorch implementation, reproducibility, debugging and modifying existing research code.
---

# implement-setting

Skill này giúp viết mã học máy **chạy đúng, chạy lại được và sửa được mà không vỡ**. Trả lời bằng tiếng Việt, giữ thuật ngữ tiếng Anh. Mục tiêu cao nhất của repo là **nhất quán**: khi có hai cách làm, chọn cách repo đang dùng (`rules.MD`).

Vai: **Professional** (cài đặt và gỡ lỗi); đầu ra qua bảng kiểm Judge ở `skill-choosing`.

## Trước khi viết bất cứ thứ gì

1. Đọc file liên quan và một file tương tự đã có. Đừng đoán tên API.
2. Tái dùng lớp cơ sở có sẵn thay vì viết mới (bảng bên dưới).
3. Làm đúng yêu cầu. Nếu không được yêu cầu, chỉ chỉnh hyperparameters trong `config/*.yaml`.
4. Ghi thay đổi vào `.log/p1.md` theo `.log/log-guide.md`; file bỏ đi chuyển vào `.bin/` thay vì xoá.

## Họ lớp của repo

| Thành phần | Lớp cơ sở | Vị trí |
|---|---|---|
| Cấu hình | `BaseConfig` (`@dataclass(frozen=True)`, có `section`) | `src/core/` |
| Model | `BaseModel` (`from_yaml`, `convert_checkpoint`, `load_pretrained` trả `LoadReport`) | `src/model/<ten>/` |
| Loss | `SupervisedLoss` / `DistillationLoss` / `SeparationLoss` | `src/loss/` |
| Trainer | `Trainer` (`configure_optimizer`, `training_step`, `evaluate`, `selection_metric`) | `src/train/` |
| Dataset | `AudioDataset` / `LabeledAudioDataset` | `scripts/load_dataset.py` |

Thuộc tính cấu hình luôn là `self.config`. Khoá yaml trùng tên field dataclass. Thư mục mã phải có `__init__.py`.

## Bốn mảng

| Việc | Đọc |
|---|---|
| Viết Dataset/DataLoader, loss, AMP, gradient accumulation, huấn luyện phân tán | `references/pytorch-patterns.md` |
| Tái lập: cấu hình, seed, môi trường, Git, resume | `references/reproducibility.md` |
| Debug: overfit một batch, shape, gradient, rò rỉ, đường cong loss | `references/debugging.md` |
| Dùng và sửa mã có sẵn, nạp checkpoint | mục bên dưới |

Chỉ đọc file khi cần, không nạp cả bốn.

## Repo đã có và chưa có gì

Biết rõ điều này để không giả định có sẵn thứ không có.

| Chủ đề | Hiện trạng |
|---|---|
| Mixed precision | Có: `amp: bf16`, autocast quanh forward, loss tính fp32 (`KDMoETrainer`). bf16 không cần `GradScaler` |
| Gradient clipping | Có: `clip_grad_norm_` trong `Trainer.run_epoch`, `grad_clip` ở config |
| Chọn batch theo VRAM | Có: `src/utils/batch_size.py::largest_fitting_batch` |
| Resume theo epoch | Có: `resume.pt` (optimizer, scheduler, EMA, best), `resume_every` |
| Seed | Có `set_seed` (random, numpy, torch, cuda); **chưa** đặt `cudnn.deterministic` hay `worker_init_fn` |
| Gradient accumulation | **Chưa có.** Xem mẫu và cảnh báo về `L_cons` trong `pytorch-patterns.md` |
| Huấn luyện đa GPU (DDP) | **Chưa có** (chỉ có xử lý tiền tố `module.` khi nạp checkpoint) |
| Hydra, Docker, conda env | **Chưa có.** Cấu hình dùng YAML + `BaseConfig`; môi trường dùng `pyproject.toml` + `requirements.txt` |
| MLflow, W&B | **Chưa có.** Kết quả ghi ở `output/` theo `RunPaths` |

Khi người dùng yêu cầu một thứ trong cột "chưa có", đề xuất cách thêm nhỏ nhất, tuân theo họ lớp ở trên, và hỏi trước nếu nó đòi thư mục cấp cao mới.

## Dùng và sửa mã có sẵn

Mục tiêu: nạp được, hiểu được, và thay đổi tối thiểu.

1. **Đọc trước khi sửa.** Tìm điểm vào (`forward`, `training_step`), nơi cấu hình được dựng, và nơi trọng số được nạp.
2. **Nạp checkpoint qua `load_pretrained`.** Nó gọi `convert_checkpoint` rồi `load_state_dict(strict=False)` và trả `LoadReport` (`missing`, `unexpected`). Kiểm tra `report.is_clean`; `strict_pretrained = True` sẽ báo lỗi nếu không sạch. Đừng viết hàm nạp trọng số riêng.
3. **Ánh xạ khoá checkpoint** trong `convert_checkpoint`. Ví dụ BEATs (`src/model/beat/beat.py`): bỏ khoá `predictor*` và các `relative_attention_bias` trùng ở layer > 0, đổi `patch_embedding.` → `patch_embed.patch_embedding.`, `.fc1.` → `.ffn.fc1.`, thêm tiền tố `backbone.`. Khi mã gốc là fairseq hoặc HuggingFace, danh sách khoá khác tên là việc phổ biến nhất.
4. **Kiểm tra hành vi sau khi nạp**, không chỉ khoá: đưa một batch qua mô hình gốc và bản đã port, so đầu ra (`torch.allclose` với dung sai). Khoá khớp nhưng thứ tự chiều sai vẫn nạp được mà cho kết quả sai.
5. **Sửa mà không phá:** thêm tham số mới bằng giá trị mặc định tái tạo hành vi cũ; thêm trường vào `NamedTuple` đầu ra ở cuối với giá trị mặc định (xem `KDMoEOutput`); chạy `pytest -m "not data"` trước và sau.
6. **Ghi lệch so với bản gốc** theo quy ước ở `doc/project-standards.md`, mục "Khác biệt FRCRN so với bài báo".

## Nguyên tắc

- Đo trước khi tối ưu: `torch.cuda.max_memory_allocated`, thời gian 50 bước đầu.
- Mọi phát biểu "đã sửa xong" phải kèm bằng chứng (test chạy, loss giảm, shape đúng), không chỉ "mã trông đúng".
- Nếu không chạy được (thiếu GPU hoặc dữ liệu), nói rõ phần nào chưa kiểm.
