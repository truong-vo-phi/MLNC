# Chạy lại 9 model pretrained trên Molab

## Vì sao phải chạy lại?

Lượt `config/config.yaml` trước dùng các constructor **scratch** của Review-3,
không nạp checkpoint bên ngoài. Áp dụng learning rate fine-tune, đặc biệt CED
`layer_decay=0.3`, vào backbone ngẫu nhiên là không phù hợp. Kết quả cũ không
đại diện cho fine-tune pretrained. Không ghi đè checkpoint cũ.

Bản này vẫn dùng `main.py` trong `imp/review-3`, nhưng gọi lại thư viện loader
pretrained đã có tại `imp/mix_dataset_v2/noise_pipeline/ce9_models.py` cùng nhánh.
Không viết lại backbone. Vì thế kiến trúc native pretrained có thể khác bản
scratch; đây không phải thí nghiệm chỉ thay trọng số trong kiến trúc cũ.

## Mở notebook

[Mở trên Molab](https://molab.marimo.io/github/truong-vo-phi/MLNC/blob/vuong_mix_v2_group_config_20261102/imp/review-3/molab.py).
Fork vào workspace của bạn. Trong **notebook specs**, bật GPU rồi chạy lần lượt:

1. **Set up**: clone nhánh, cài thư viện phụ và kiểm tra GPU. Giữ PyTorch/CUDA
   có sẵn của Molab; không hạ cấp theo môi trường Windows.
2. **Download**: lấy bản dataset đã ghim revision, kiểm tra SHA256, chỉ giải nén
   `train/eval` mixture gốc. Không dùng cache FRCRN, không đọc tập test. Chuẩn bị
   khoảng 25–30 GB đĩa cho dữ liệu, tải trọng số và checkpoint.
3. **Check pretrained**: kiểm tra cả 9 model trên CPU, logits 36 lớp hữu hạn;
   không train. Đọc `logs/pretrained/preflight.json` và thông báo VERIFIED.
4. **Authorize Drive**: đăng nhập tài khoản của bạn qua hướng dẫn trong cell.
   Không gửi token, mật khẩu hay URL có token vào chat.
5. **Train**: mặc định BEATs; chọn `all` để chạy tuần tự cả 9. Giữ nguyên tên
   thư mục Drive và lựa chọn model khi muốn resume. Nếu đã chạy riêng BEATs,
   chọn từng model còn lại; không đổi lượt riêng đó sang `all` trong cùng thư mục
   backup vì snapshot config của queue khác. Bấm **Refresh training log** để xem tiến trình.

Notebook chỉ bắt đầu training khi bấm nút 5. Phiên Molab có giới hạn thời gian;
chạy nền không vượt được giới hạn phiên. File do chương trình sinh ra không
được coi là đã lưu bền vững chỉ vì còn thấy trong đĩa phiên.

## Trọng số và cấu hình

| Model | Nguồn khởi tạo backbone |
|---|---|
| PANN-CNN10 | AudioSet, checkpoint mAP 0.380 |
| CED Tiny / Mini / Small | `mispeech/ced-*`, revision cố định |
| EfficientAT MN10 | AudioSet, `mn10_as_mAP_471.pt` |
| ViT-S | timm AugReg ImageNet-21k → ImageNet-1k |
| MobileViT-S / MobileNetV3-Large | timm ImageNet-1k |
| BEATs | iter3+ AS2M fine-tuned cpt2, SHA256 cố định |

BEATs tải qua mirror Hugging Face ghim revision, xác nhận nội dung bằng SHA256
khớp checkpoint Microsoft đã dùng trước đây. Chi tiết tại
`config/pretrained_sources.json`. Head 36 lớp được khởi tạo mới, backbone mở
toàn bộ. Vision pretrained là ImageNet, không phải AudioSet.

`config/pretrained.yaml` + `config/models.yaml`: giữ hyperparameter nhóm, native
frontend phù hợp pretrained, thêm crop/pad frame và SpecAugment theo profile.
50 epoch tối đa, patience 8, smoothing 0.1, effective batch 64, EMA được theo dõi,
chọn checkpoint theo `macro_acc` của nhóm. LR BEATs 1e-5/1e-4; ViT/Mobile
3e-5/3e-4 cho backbone/head. CED vẫn giữ layer decay 0.3 theo nhóm.

## Kết quả và resume

- Local phiên: `checkpoints/pretrained/<model>/best.pt`, `last.pt`,
  `history.csv`, `results.json`, `metrics_per_snr.csv`, `effective_config.json`.
- Có train_loss, train_acc, train_mAP, val_loss, val_acc, val_mAP từng epoch;
  acc, precision, recall, macro_f1, micro_f1, mAP cho all và 6 mốc SNR.
- Drive: `<thư mục đã chọn>/<model>/`. Best được tải lên khi cải thiện;
  cặp checkpoint resume tại epoch 1, mỗi 5 epoch và khi hoàn tất. Có thể mất
  tối đa 4 epoch từ lần backup định kỳ; kiểm tra `backup_status.json` và log.
- Sau khi phiên hết hạn: mở lại notebook, chạy bước 1–4 rồi chọn cùng model,
  cùng thư mục Drive và bấm train. Cặp last/best được tải về, xác minh SHA256,
  khôi phục optimizer/scheduler/RNG. Config khác sẽ bị từ chối resume.

Lệnh tương đương sau khi đã đăng nhập Drive:

```bash
cd /tmp/mlnc-review3/MLNC/imp/review-3
python main.py prepare --remove-archive
python main.py preflight --model all --device cpu
python main.py train --model beats --config config/pretrained.yaml --device cuda --drive-dir MLNC/review3_pretrained_v2
```

Nếu notebook lỗi ở bước nào, dừng tại bước đó và gửi log lỗi đã bỏ thông tin
đăng nhập. Chưa có kết quả training mới chỉ từ việc chạy preflight.

Kiểm tra local ngày 03/11/2026: cả 9 loader pretrained và forward trên CPU
đều thành công; 44 test CPU pass, gồm kiểm tra backup/resume và checksum.
Chưa chạy training mới trên Molab; cần đăng nhập và bật GPU trong notebook.

## Tài liệu nền tảng

- [Molab: GitHub notebook, bật GPU và giới hạn phiên](https://docs.marimo.io/guides/molab/).
- [Lưu trữ Molab và sao lưu sang Drive](https://molab.marimo.io/blog/seamless-storage-in-molab).
