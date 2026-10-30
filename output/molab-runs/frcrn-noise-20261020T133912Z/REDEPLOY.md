# Tái triển khai lần chạy FRCRN (target noise) lên molab

Hồ sơ lưu lúc 2026-10-20T13:39:13+00:00. Sandbox ghi nhận: `https://sb-414bc7f28e9e88dc.sb.molab.run/`.

## Vị trí trên Google Drive
- Thư mục: `FA26_Capstone/Truong/frcrn-snr-experts`
- Hồ sơ lần chạy (tiến độ, mốc thời gian từng epoch): `FA26_Capstone/Truong/frcrn-snr-experts/run_manifest.json`
- Bản tốt nhất mới nhất: `FA26_Capstone/Truong/frcrn-snr-experts/snr20/best.pt`
- Bản lịch sử có mốc thời gian: `FA26_Capstone/Truong/frcrn-snr-experts/snr20/best_e<epoch>_<UTC>.pt`
- Trạng thái để nối tiếp (trọng số, optimizer, lịch sử epoch): `FA26_Capstone/Truong/frcrn-snr-experts/snr20/resume.pt`
- Kết quả cuối (chỉ có khi chạy xong): `FA26_Capstone/Truong/frcrn-snr-experts/snr20/result.json`

## Code đã dùng
- Commit local: `13316c45b31a51db6757ffeecb34f02227820a85` (64 tệp chưa commit); mã băm code: `fbc01dd03ec3db1195531dd17d9d4eaf`.
- Bản đóng gói đúng các tệp đã dùng (kèm bảng offset): `D:/PROJECT/MLNC/output/molab-runs/frcrn-noise-20261020T133912Z/code.tar.gz`.

## Các bước
1. Tạo sandbox molab mới, mở notebook, chạy cell đăng nhập Google Drive (token được lưu trên sandbox).
2. Trên máy này:
   ```bash
   export MARIMO_TOKEN=<token của sandbox mới>
   python -m scripts.molab_deploy push --url <URL sandbox mới> --archive D:/PROJECT/MLNC/output/molab-runs/frcrn-noise-20261020T133912Z/code.tar.gz
   export KAGGLE_ACCESS_TOKEN=<token Kaggle>   # chỉ cần nếu sandbox chưa có dữ liệu
   python -m scripts.molab_deploy launch --url <URL sandbox mới> --kaggle -- --levels 20 --use-drive --drive-dir FA26_Capstone/Truong/frcrn-snr-experts
   python -m scripts.molab_deploy status --url <URL sandbox mới>
   ```
3. Cùng thư mục Drive nên script tự khôi phục `resume.pt`/`best.pt` và nối tiếp từ epoch đã xong. Nếu `result.json` đã có
   thì lần chạy coi như xong; muốn chạy lại từ đầu hãy dùng thư mục Drive mới.
