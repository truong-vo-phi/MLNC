# Toán và ML cơ bản: vì sao một kỹ thuật hoạt động

Mỗi mục theo khung: vấn đề, công thức, vì sao hoạt động, khi sai thì thấy gì, nằm đâu trong repo.

## Mục lục
1. KL và chưng cất nhiệt độ
2. EMA và Mean Teacher
3. Load-balancing và entropy loss cho router
4. Label smoothing
5. L1 bất đối xứng
6. Consistency bằng cosine
7. SI-SDR
8. Cross-fit (out-of-fold)
9. Tối ưu hoá: AdamW, warmup, cosine, grad clip, bf16
10. Mẹo chẩn đoán

## 1. KL và chưng cất nhiệt độ

- **Vấn đề:** nhãn cứng chỉ nói "lớp nào đúng", teacher còn biết lớp nào *giống* lớp nào.
- **Công thức:** `L = T² · KL( softmax(z_t/T) ‖ softmax(z_s/T) )`, với `KL(p‖q) = Σ p·(log p − log q)`.
- **Vì sao hoạt động:** T > 1 làm phân phối phẳng hơn, phần đuôi (xác suất nhỏ nhưng có thông tin) đủ lớn để cho gradient. Gradient của số hạng mềm co theo 1/T², nên nhân T² giữ cho đóng góp của nó ngang với CE khi đổi T (Hinton 2015).
- **Khi sai:** quên T² thì đổi T làm KD yếu đi mà bạn tưởng nó "không giúp"; quên `detach` trên teacher thì gradient chảy vào teacher; KL không đối xứng nên đổi chỗ `p` và `q` cho loss khác.
- **Repo:** `src/loss/kdmoe.py::router_kl`, `T = 2` trong `kd_temperature`. Teacher cache logit 36 lớp, `p_t = softmax(z_t/T)·M` gộp thành 4 nhóm, nên T có thể quét mà không chạy lại teacher.

## 2. EMA và Mean Teacher

- **Vấn đề:** trọng số SGD dao động quanh nghiệm; checkpoint một thời điểm nhiễu.
- **Công thức:** `θ_ema ← d·θ_ema + (1−d)·θ`. Cửa sổ hiệu dụng ≈ 1/(1−d); `d = 0,999` là ≈ 1000 bước.
- **Vì sao hoạt động:** trung bình theo thời gian làm mượt nhiễu của bước cập nhật, thường rơi vào vùng phẳng hơn của loss nên tổng quát tốt hơn. Trong Mean Teacher, bản EMA đóng vai teacher cho consistency loss, cho mục tiêu ổn định hơn chính student.
- **Khi sai:** `d` quá lớn so với số bước thì EMA còn mang trọng số khởi tạo; đánh giá bằng trọng số thường thay vì EMA (hoặc ngược lại) cho kết quả khác. Với BatchNorm phải xử lý cả buffer.
- **Repo:** `src/utils/ema.py`; trainer `KDMoETrainer` đánh giá qua `ema.swap`, `ema_decay = 0,999`.

## 3. Load-balancing và entropy loss cho router

- **Vấn đề:** router top-k dễ sập: vài expert nhận hết token, phần còn lại không học (rich-get-richer).
- **Công thức (Switch):** `L_aux = α·N·Σ_i f_i·P_i`, với `f_i` là tỉ lệ token *thực sự* gửi đến expert i (rời rạc), `P_i` là xác suất router trung bình dành cho i. Cực tiểu khi cả hai đều đều 1/N, khi đó `L_aux = α`.
- **Vì sao hoạt động:** `f_i` không có gradient nhưng `P_i` có; tích `f_i·P_i` phạt việc dồn xác suất vào expert đang bận, nên gradient đẩy `P_i` của expert bận xuống.
- **Biến thể:** mean-importance của SpeechMoE2 `L_m = n·Σ_i ( (1/k)·Σ_j p_ij )²`; entropy loss giảm entropy mỗi token để quyết định dứt khoát, đi kèm cân bằng để các expert vẫn được dùng.
- **Khi sai:** α quá lớn ép phân phối đều, phá chuyên môn hoá; quá nhỏ thì sập. Theo dõi entropy router và tỉ lệ token mỗi expert.
- **Repo:** KD-MoE là **soft** MoE (cả 4 expert chạy, trọng số mềm) và router được giám sát bằng KD, nên không có số hạng cân bằng; `route_ce_weight` là tuỳ chọn giám sát nhóm bằng nhãn.

## 4. Label smoothing

- **Công thức:** đích `y' = (1−ε)·onehot + ε/K`, `ε = 0,1`, `K = 36`.
- **Vì sao hoạt động:** CE với one-hot đẩy logit đúng ra vô cùng; làm mềm đích giữ logit hữu hạn, mô hình bớt quá tự tin và hiệu chỉnh (calibration) tốt hơn.
- **Khi sai:** làm hỏng KD nếu đứng trước teacher đã làm mềm; làm yếu so sánh độ tin cậy giữa lớp. Không dùng khi cần xác suất đúng nghĩa mà chưa hiệu chỉnh.
- **Repo:** `label_smoothing = 0,1` ở `L_CE` của student; BEATs fine-tune để 0.

## 5. L1 bất đối xứng (L_front)

- **Công thức:** `w = 1 + (λ−1)·1[ŷ − y > 0]`, `L = mean(w·|ŷ − y|)`, `λ = 2`.
- **Vì sao hoạt động:** `ŷ > y` nghĩa là ước lượng nhiễu *lớn hơn* nhiễu thật, tức phần tiếng nói bị tính vào nhiễu. Phạt phần thừa gấp λ lần ép bộ lọc thà bỏ sót nhiễu còn hơn để rò tiếng nói, vì tiếng nói lẫn vào làm hỏng phân loại nhiễu. L1 chịu ngoại lai tốt hơn L2.
- **Khi sai:** λ quá lớn dẫn đến ước lượng nhiễu quá nhỏ, mask co về 0.
- **Repo:** `src/loss/kdmoe.py::FrontLoss`, `front_lam_over`.

## 6. Consistency bằng cosine

- **Công thức:** `L_cons = 1 − cos(e_i, e_j)`, cặp (i, j) cùng clip nhiễu, khác tiếng nói hoặc SNR.
- **Vì sao hoạt động:** buộc embedding phụ thuộc vào *danh tính nhiễu* chứ không vào tiếng nói hay mức SNR (bất biến với yếu tố gây nhiễu). Cosine bỏ qua độ lớn nên không ép norm.
- **Khi sai:** nếu cặp không thật sự cùng nhiễu (sampler lỗi) thì loss dạy ngược; trọng số quá lớn làm embedding sập về một điểm.
- **Repo:** `cons_weight = 0,1`, cặp sinh bởi `PairedClipSampler` với khoá `noise_ytid`.

## 7. SI-SDR

- **Công thức:** `α = ⟨ŝ, s⟩ / ‖s‖²`, `SI-SDR = 10·log10( ‖α·s‖² / ‖α·s − ŝ‖² )`.
- **Vì sao dùng:** bất biến theo scale của ước lượng, nên không bị thưởng/phạt vì âm lượng; đo đúng chất lượng dạng sóng.
- **Chú ý với nhiễu:** khi SNR cao nhiễu rất nhỏ, SI-SDR của nhiễu ước lượng thấp (repo: 13,5 dB ở −5 dB xuống 0,86 dB ở 20 dB) là dấu hiệu bài toán khó, không hẳn lỗi mô hình.
- **Repo:** `src/loss/frcrn.py::si_snr`.

## 8. Cross-fit (out-of-fold)

- **Vấn đề:** probe huấn luyện rồi dự đoán chính những mẫu đó thì quá tự tin, KD dạy student sự tự tin giả.
- **Cách làm:** chia k fold theo nhóm (ở đây theo `noise_clip_id` để hai mix cùng nhiễu không rò qua fold), dự đoán fold bị giữ lại.
- **Khi sai:** chia theo mẫu thay vì theo nhóm gây rò; so sánh với OOF accuracy mới là con số trung thực.
- **Repo:** `scripts/teacher_cache.py`, `StratifiedGroupKFold`.

## 9. Tối ưu hoá

- **AdamW:** tách weight decay khỏi cập nhật thích nghi; `wd = 0,05`.
- **Warmup + cosine:** warmup tránh bước lớn khi moment ước lượng chưa ổn; cosine hạ lr mượt về `min_lr`.
- **Grad clip:** chặn norm gradient, quan trọng khi có GRU và bf16.
- **bf16:** cùng dải mũ với fp32 nhưng ít bit mantissa; tính loss, softmax, KL, log ở fp32 để tránh NaN khi nhiễu gần 0 (đã ghi trong `doc/moe.md`).

## 10. Mẹo chẩn đoán

| Triệu chứng | Nghĩ đến |
|---|---|
| KD loss không giảm | quên `detach`, T sai, `p_t` không chuẩn hoá về 1 theo nhóm |
| Router acc ≈ 25% (4 nhóm) | router chưa học; kiểm tra gradient vào `fc2`, `kd_weight` |
| Mask `g` ≈ 0 hoặc ≈ 1 | `L_front` quá mạnh hoặc yếu; xem `gate_mean`, `gate_low` |
| Val tốt hơn train sớm | regularization (dropout, drop-path, SpecAugment) đang mạnh; bình thường |
| Accuracy giảm khi SNR tăng | không phải lỗi: tiếng nói lấn nhiễu (xem `audio-snr.md`) |
