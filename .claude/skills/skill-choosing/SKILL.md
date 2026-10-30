---
name: skill-choosing
description: Chọn skill phù hợp rồi triển khai ngay trong repo MLNC. Phân tích bối cảnh hiện tại dựa trên các skill sẵn có (research-platform, implement-setting, experiment-setting, soft-skill, marimo-pair và các skill khác), nếu không skill nào khớp thì hỏi kỹ người dùng; đưa ra danh sách skill cần dùng để giải quyết vấn đề; sau khi xác định được skill thì lập tức triển khai. Không được chạy trên máy local (huấn luyện, đánh giá, tải dữ liệu lớn, notebook) khi chưa có sự cho phép của người dùng. Hãy dùng skill này khi yêu cầu của người dùng rộng, mơ hồ, trộn nhiều việc (ví dụ "giúp tôi cải thiện kết quả", "bắt đầu thí nghiệm này", "xem bài báo rồi cài đặt thử"), hoặc khi chưa rõ nên dùng skill nào, kể cả khi họ không nhắc tên skill. Mỗi việc đi qua ba vai Researcher (thu bằng chứng), Professional (làm việc chuyên môn) và Judge (kiểm trước khi báo). Use to route a request to the right skill(s) with Researcher, Professional and Judge roles, then execute without running anything locally unless approved.
---

# skill-choosing

Skill này là **bộ định tuyến**: đọc bối cảnh, chọn skill, rồi làm. Nó không thay thế các skill chuyên môn. Trả lời bằng tiếng Việt.

## Quy trình

1. **Đọc bối cảnh.** Gồm yêu cầu hiện tại, các tệp đang mở hoặc vừa nhắc, tiến trình đã làm trong cuộc trò chuyện, và trạng thái repo. Chỉ đọc (không chạy gì).
2. **Đối chiếu với danh sách skill có sẵn** trong phiên (bảng bên dưới và mọi skill khác được liệt kê trong ngữ cảnh). Nếu không chắc tên, dùng công cụ liệt kê skill thay vì đoán.
3. **Chọn skill.** Mỗi việc một skill chính; thêm skill phụ chỉ khi nó thật sự đóng góp. Nêu một dòng lý do.
4. **Nếu không skill nào khớp, hỏi kỹ người dùng** (mục "Khi không chắc"). Đừng tự điền giả định lớn.
5. **Triển khai ngay** khi đã rõ: gọi skill đã chọn và làm theo hướng dẫn của nó. Đừng dừng lại để xin phép cho những việc chỉ đọc hoặc chỉnh sửa tệp.
6. **Qua Judge** (mục "Ba vai") rồi **báo cáo ngắn:** đã chọn skill nào, đã làm gì, còn gì chưa làm.

## Ranh giới "chạy trên local"

Đây là ràng buộc cứng của skill này.

| Được làm không cần hỏi | Phải có sự cho phép trước |
|---|---|
| Đọc, tìm kiếm tệp; đọc log, kết quả đã có | Chạy huấn luyện, fine-tune, đánh giá |
| Viết, sửa tệp mã, cấu hình, tài liệu | Chạy notebook hoặc cell tính toán nặng |
| Phân tích tĩnh, lập kế hoạch, ước lượng | Tải dataset hoặc checkpoint lớn |
| Tra cứu web, đọc bài báo | Chạy `pytest` toàn bộ hoặc script dùng GPU/CPU nặng |
| | Bất kỳ lệnh nào ghi đè hoặc xoá `output/`, `datasets/`, `pretrained/` |

Khi cần chạy: nêu **lệnh chính xác, nơi chạy (local hay molab), ước lượng thời gian và tài nguyên, và cái nó ghi ra**, rồi chờ người dùng đồng ý. Một lần đồng ý chỉ áp dụng cho lệnh đó.

Khi tiến trình đang chạy ở nơi khác (ví dụ notebook molab), **không làm gián đoạn nó**: ưu tiên đọc log và tệp kết quả; chỉ chạy mã trong kernel đang bận khi chắc chắn không ảnh hưởng.

## Bảng định tuyến

| Việc người dùng cần | Skill chính | Skill phụ |
|---|---|---|
| Đọc, tóm tắt, tái hiện bài báo; đối chiếu công thức với mã; giải thích loss hoặc kỹ thuật; hỏi về SNR, mel, nhiễu | `research-platform` | `soft-skill` (tiếng Anh học thuật) |
| Viết hoặc sửa model, loss, trainer, dataset; nạp checkpoint; NaN, loss không giảm; tái lập; DDP, AMP | `implement-setting` | `research-platform` (hiểu cơ chế) |
| Lập kế hoạch ablation, chọn seed và metric, phân tích lỗi, ước lượng thời gian và VRAM | `experiment-setting` | `implement-setting` (hiện thực) |
| Viết báo cáo, README, abstract; trình bày kết quả âm; nghi ngờ một con số; ưu tiên việc; tiếng Anh | `soft-skill` | `experiment-setting` |
| "Bước tiếp theo là gì", chọn hướng đi dựa trên kết quả hiện có, ưu tiên thí nghiệm khi GPU có hạn | `research-next-step` | `experiment-setting` |
| Bài báo dùng dataset nào, xử lý dữ liệu ra sao, train/test trên bộ nào | `research-platform` (chế độ A, bước "Dữ liệu và giao thức") | `artifact-diagramming` chỉ khi người dùng xin vẽ pipeline |
| Viết hoặc điền chương báo cáo capstone (`latex/capstone/`) | `soft-skill` (chế độ A, `writing.md` §7) | `research-platform` (trích dẫn), `experiment-setting` (bảng kết quả) |
| Giải thích cấu trúc mô hình của một bài báo (từ lớn đến nhỏ), đối chiếu với mã trong repo | `research-platform` (chế độ A), trả lời đầy đủ trong chat | `artifact-diagramming` chỉ khi người dùng yêu cầu hình, và sau khi đã trả lời xong |
| Làm việc cùng notebook marimo đang chạy | `marimo-pair` | `implement-setting` |
| Nhìn lại một phiên marimo đã xong để cải tiến tài liệu skill | `retro-marimo-pair` | |
| Một yêu cầu chạm nhiều lĩnh vực cùng lúc | Chia thành các việc nhỏ, mỗi việc một skill (xem mục "trộn nhiều việc") | |
| Xem lại thay đổi mã đang chờ commit | `code-review` (lỗi đúng sai) hoặc `simplify` (làm gọn) | `implement-setting` |

### Skill tích hợp và skill của Claude Code

Khi việc đúng là loại dưới đây, dùng thẳng skill tương ứng thay vì viết tay:

| Việc | Skill |
|---|---|
| Xuất báo cáo/bài viết ra Word, PDF, Excel, PowerPoint | `anthropic-skills:docx`, `pdf`, `xlsx`, `pptx` |
| Soạn tài liệu để chia sẻ, bình luận (không chỉ định định dạng file) | `anthropic-skills:docs` |
| Biểu đồ, dashboard, hình kết quả (đường cong loss, confusion, so sánh model) | `dataviz` |
| Trang HTML chia sẻ được (Artifact) | `artifact-design`, thêm `artifact-capabilities` hoặc `artifact-diagramming` khi cần |
| Khảo sát nhiều nguồn rồi tổng hợp thành báo cáo | `anthropic-skills:deep-research` |
| Chạy lặp hoặc lên lịch | `loop`, `schedule` |
| Chỉnh cấu hình Claude Code (settings, hooks, quyền) | `update-config`, `fewer-permission-prompts` |

Trùng chức năng: `anthropic-skills:research-next-steps` là bản chung của `research-next-step`; trong repo này luôn chọn `research-next-step` (đã biết đường dẫn kết quả, ràng buộc molab và `rules.MD`).

Quy tắc ranh giới vẫn áp dụng: dùng các skill này để **viết, xuất, vẽ từ số liệu đã có**, không để chạy huấn luyện hay đánh giá.

## Khi một yêu cầu trộn nhiều việc

Tách thành chuỗi theo thứ tự phụ thuộc, nêu rõ. Ví dụ "đọc bài này rồi thử cài vào repo":

1. `research-platform`: đọc bài, tách cốt lõi, đối chiếu công thức với mã hiện có.
2. `implement-setting`: viết thay đổi nhỏ nhất theo họ lớp của repo.
3. `experiment-setting`: đề xuất ablation và ước lượng chi phí.
4. **Dừng và xin phép** trước khi chạy bất cứ thứ gì.

## Ba vai: Researcher, Professional, Judge

Mỗi việc đi qua một vòng ba vai, do cùng một agent đảm nhận lần lượt. Không tạo skill mới: các skill sẵn có đã phủ từng vai.

| Vai | Làm gì | Dùng skill | Không được |
|---|---|---|---|
| Researcher | Thu bằng chứng trước: đọc nguồn, log, mã, bài báo; gắn nhãn mỗi tuyên bố là **đọc từ nguồn**, **suy từ mã** hay **giả thuyết** | `research-platform`, `research-next-step` | Kết luận hoặc sửa mã |
| Professional | Làm việc chuyên môn theo bằng chứng đã có: viết mã, cấu hình, kế hoạch thí nghiệm, báo cáo | `implement-setting`, `experiment-setting`, `soft-skill` | Vượt `rules.MD` mục 0 hoặc chạy khi chưa được phép |
| Judge | Kiểm bản nháp theo bảng dưới **trước khi báo cáo**; ra phán quyết đạt hoặc không đạt | `soft-skill` mục C, công cụ `advisor` nếu có | Báo cáo vòng qua một mục không đạt |

Bảng kiểm của Judge (mỗi mục đạt hoặc không đạt, kèm một dòng bằng chứng):

1. Mỗi con số truy được về tệp hoặc nguồn, có cấu hình và split đi kèm.
2. Không khẳng định điều nguồn không nói (chi tiết chưa đọc được thì ghi "chưa kiểm chứng").
3. Không chạy gì ngoài phạm vi được cho phép; sửa mã không vượt `rules.MD`.
4. Nội dung chính nằm đầy đủ trong chat; hình chỉ thêm sau.
5. Việc sửa tệp đã ghi vào `.log/p1.md`; tệp bỏ đi nằm trong `.bin/`.
6. Con số quá tốt hoặc bất ngờ đã qua thứ tự kiểm của `soft-skill` mục C.

Phán quyết: **đạt** thì báo cáo; **không đạt** thì trả lại đúng vai gây lỗi (thiếu nguồn về Researcher, sai việc về Professional) và sửa rồi kiểm lại. Ví dụ đã xảy ra: tuyên bố "mixture được trộn ngay khi huấn luyện" không có nguồn trong bài báo; Judge phải bắt và rút lại trước khi báo.

Judge có thể gọi `advisor` khi công cụ này có. Không tự ý khởi chạy subagent nếu người dùng không yêu cầu.

## Khi không chắc

Hỏi ít nhưng đúng: tối đa vài câu, mỗi câu có thể trả lời nhanh, và nêu mặc định bạn sẽ chọn nếu họ không trả lời.

- Mục tiêu cuối là gì (hiểu, cài đặt, báo cáo, chạy)?
- Dữ liệu và cấu hình nào là bối cảnh (S/M/L, split, SNR)?
- Được chạy ở đâu (chỉ viết mã, local, hay molab)?
- Cái gì **không** được đụng (tiến trình đang chạy, thư mục `output/`)?

## Hình thức trả lời

Người dùng cần **đọc được thông tin ngay trong chat**. Hình chỉ là phần thêm.

- Nội dung chính (cấu trúc, số liệu, so sánh, kết luận) viết đầy đủ trong chat, trước khi làm bất cứ thứ gì khác. Không đẩy nội dung chính sang Artifact hay trang ngoài rồi chỉ trả về một đường dẫn.
- Giải thích cấu trúc thì đi từ lớn đến nhỏ: hệ thống, thành phần, khối, lớp, rồi tới chi tiết. Dùng danh sách phân cấp và bảng ngắn cho tên lớp, kênh, kích thước, tham số.
- Chỉ vẽ canvas, Artifact hoặc sơ đồ khi người dùng yêu cầu rõ, và làm **sau** khi đã trả lời đủ trong chat. Nếu họ chỉ nói "minh hoạ" mà chưa nói rõ, trả lời trong chat trước, rồi hỏi một câu có muốn thêm trang minh hoạ không.
- Khi vẽ: một hình một ý, ít màu, nhãn ngắn và rõ, đủ khoảng trống giữa các khối. Nếu không chắc hình sẽ gọn và đẹp, dùng bảng hoặc danh sách thay cho hình, và nói rõ đã chọn như vậy.
- Gắn nhãn mức chắc chắn cho từng con số: đọc từ nguồn, suy từ mã, hay tự tính. Phần chưa kiểm chứng nói rõ là chưa kiểm chứng.
- Hỏi trước khi xuất bản trang ra ngoài chat, vì trang đó sinh ra một liên kết mà người dùng phải mở.

## Nguyên tắc

- Chọn skill bằng bằng chứng trong bối cảnh, không bằng từ khoá đơn lẻ.
- Không tự mở rộng phạm vi ngoài yêu cầu (`rules.MD` mục 0: nếu không được yêu cầu, chỉ tuỳ chỉnh hyperparameters).
- Việc sửa mã hoặc cấu hình phải ghi vào `.log/p1.md` theo `.log/log-guide.md`; nhắc việc này khi chuyển sang `implement-setting`.
- File bỏ đi đưa vào `.bin/`, không xoá thẳng (`rules.MD`).
- Khi đã chọn xong, làm tiếp; đừng dừng để giải thích việc chọn quá dài.
