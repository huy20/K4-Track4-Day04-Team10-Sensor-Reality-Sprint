# Pitch Bước 6 · Nhóm 10 (mục tiêu 4 phút, tối đa 5)

Thứ tự: Problem → Method → Benchmark → Failure → Decision. Mỗi phần ghi thứ cần mở trên màn hình và lời nói gợi ý. Nói theo ý, không cần thuộc từng chữ. Báo cáo đầy đủ: [`step6_report.md`](step6_report.md).

| Thời gian | Phần | Mở trên màn hình |
|---|---|---|
| 0:00–0:35 | Problem | Mục 1 của báo cáo |
| 0:35–1:20 | Method | Bảng Method và khối Truy vết |
| 1:20–2:30 | Benchmark | [`results/main_figure.png`](../results/main_figure.png) |
| 2:30–3:15 | Failure case | [`results/failure_case_blur_sigma5p5_compact.png`](../results/failure_case_blur_sigma5p5_compact.png) |
| 3:15–4:15 | Decision | Bảng luật cảnh báo ở mục 5 |

## Lời nói

### 1. Problem (0:00–0:35)

Nhóm 10 làm bài toán camera trước của xe có ADAS. Tính năng là phát hiện xe, xe máy, người đi đường để cảnh báo va chạm. Ngoài đường, camera hỏng theo những cách rất đời thường: ống kính bẩn hay ướt làm ảnh nhòe, xe rung làm ảnh kéo vệt, vào hầm thì tối, nắng gắt thì cháy sáng, ban đêm thì nhiễu. Nguy hiểm nhất là detector bỏ sót vật mà không báo lỗi. Nhóm hỏi hai câu. Một: detector tụt bao nhiêu với từng loại lỗi? Hai: một health score tính từ chính ảnh có báo trước được không?

### 2. Method (0:35–1:20)

Detector là YOLOv8n của Ultralytics, nhóm train trên bộ ảnh đường phố 15 lớp lấy từ Roboflow. Cách đo theo paper của Michaelis và cộng sự năm 2019: giữ nguyên ảnh và nhãn, chỉ làm hỏng ảnh theo 5 loại lỗi, mỗi loại 4 mức. Paper kết luận các detector chuẩn chỉ còn 30 đến 60% hiệu năng khi ảnh bị hỏng. Đó là kết luận của paper, không phải số của nhóm. Song song, nhóm tính health score từ 3 đại lượng: độ nét bằng variance of Laplacian, độ sáng và entropy. Health score chỉ cần OpenCV, không cần GPU.

### 3. Benchmark (1:20–2:30)

Đây là số nhóm tự đo trên 200 ảnh thật; lỗi là mô phỏng. Baseline là chính 200 ảnh đó khi chưa bị làm hỏng, mAP@50 bằng 0.466. Đường xanh là detector, đường cam là health score, cùng tính theo phần trăm so với baseline. Có ba điểm.

- Một: noise nặng nhất, mAP giảm 82%. Hai loại blur giảm hơn 50%. Cháy sáng chỉ giảm 9%.
- Hai: với blur và thiếu sáng, health giảm cùng detector, nghĩa là báo được.
- Ba, quan trọng nhất: với noise, detector gần như mù nhưng health vẫn 100%. Nhiễu tạo cạnh giả nên variance of Laplacian tăng từ 616 lên 9475, và health tưởng ảnh nét hơn.

So với paper: hai loại blur còn khoảng 45–48% hiệu năng, khớp khoảng 30–60%; noise chỉ còn 18%, tệ hơn.

### 4. Failure case (2:30–3:15)

Failure case nhóm phân tích là blur sigma 5.5, giống ống kính bẩn hoặc lệch nét. Bên trái, detector thấy xe máy, người lái, xe tải với confidence 0.91 đến 0.93. Bên phải, cùng khung hình sau khi nhòe, không còn box nào. Trên 200 ảnh, recall giảm từ 0.41 xuống 0.21. Nguyên nhân: blur xóa cạnh, nên vật nhỏ và mảnh như xe máy mất đặc trưng trước. Với cảnh báo va chạm, đây là bỏ sót im lặng. Nhìn confidence cũng không thấy lỗi, vì box còn sót vẫn có confidence 0.89. Health score thì thấy: giảm từ 0.93 xuống 0.61.

### 5. Engineering decision (3:15–4:15)

Từ hai quan sát đó, nhóm quyết định không dùng health score một mình. Health bắt được blur nhưng mù với noise; confidence thì ngược lại. Nhóm thử trên 20 điều kiện lỗi. Health dưới 0.70 bắt được 6 trên 9 điều kiện làm mAP tụt hơn 20%, và sót cả 3 mức noise. Thêm điều kiện confidence trung bình dưới 0.85 thì bắt đủ 9 trên 9, báo nhầm 3 trên 11. Ngưỡng này chọn trên chính dữ liệu đó nên còn lạc quan. Khi kích hoạt: cảnh báo tài xế, hạ mức hỗ trợ, dùng radar để phanh.

Về trade-off: health score rẻ, dễ giải thích, hợp làm lớp kiểm tra đầu tiên. Nhưng ở cảnh ít chi tiết như cao tốc vắng, sương mù, hay drone nhìn lên trời, nó sẽ báo nhầm.

Bước tiếp theo:
- ghi log từng khung hình;
- lấy ảnh lỗi thật có nhãn, ví dụ BDD100K ban đêm và trời mưa;
- train lại với augmentation blur và noise, rồi chạy lại đúng benchmark này.

Cảm ơn mọi người đã lắng nghe.

## Khi bị hỏi "nhóm đã chạy gì?": mở ngay

| Câu hỏi | Mở |
|---|---|
| Có chạy thật trên GPU không? | Cell cài đặt đầu notebook (log `ultralytics.checks()`: `Tesla T4`) và mục 9 (log `val()` từng mức: `all 200 1183 …`) |
| Ảnh lỗi trông thế nào? | Notebook mục 4 (lưới 5 loại lỗi × 5 mức) |
| Health tính thế nào? | Notebook mục 5, hàm `health_scores` |
| Số trong bảng lấy từ đâu? | [`results/benchmark_table.csv`](../results/benchmark_table.csv), lấy từ notebook mục 9. Tạo lại: `python scripts/make_step6_figures.py` |
| Luật fallback 9/9 kiểm ở đâu? | [`results/fallback_rule_check.csv`](../results/fallback_rule_check.csv) |
| Có đủ 4 ví dụ failure không? | [`results/failure_case_blur_sigma5p5.png`](../results/failure_case_blur_sigma5p5.png) (notebook mục 11) |

## Câu hỏi dễ gặp

- **Vì sao baseline là 0.466 mà notebook ghi 0.362?** Số 0.362 đo trên cả 1980 ảnh valid. Mọi mức lỗi chỉ chạy trên 200 ảnh đầu, nên phải so cùng tập ảnh: mức 0 trên 200 ảnh là 0.466. Chênh lệch này cho thấy 200 ảnh đầu dễ hơn trung bình. Đó là giới hạn, nhóm đã ghi trong báo cáo.
- **Vì sao blur nhẹ lại làm mAP tăng khoảng 12%?** Precision tăng (0.826 → 0.881), recall giảm nhẹ (0.414 → 0.394). Có thể làm mượt đã bớt box sai do nhiễu nén ảnh. Nhưng với 200 ảnh, chạy 1 lần, không có khoảng tin cậy thì chưa kết luận được. Cần bootstrap hoặc chạy cả tập valid.
- **Vì sao health không thấy noise?** Nhiễu làm variance of Laplacian tăng (616 → 9475) nên điểm độ nét bị chặn ở 1. Entropy cũng tăng nên điểm entropy cũng là 1. Chỉ điểm phơi sáng đổi chút ít, nên health vẫn khoảng 0.94. Cách sửa: thêm bước ước lượng mức nhiễu trên vùng ảnh phẳng, đọc gain/ISO từ ISP, hoặc dùng tín hiệu confidence như luật kết hợp.
- **Notebook mục 10 có tương quan health–mAP@50 chỉ 0.23?** Đúng, tổng thể rất yếu, vì hai nhóm lỗi đi ngược chiều: blur làm health giảm, noise làm health tăng. Phải tách theo từng loại lỗi như plot chính mới thấy rõ.
- **Vì sao thiếu sáng và cháy sáng ảnh hưởng ít?** Giả thuyết: augmentation mặc định của Ultralytics có đổi độ sáng (`hsv_v = 0.4`) nhưng gần như không có blur hay noise, nên model đã quen với thay đổi độ sáng. `train.py` dùng augmentation mặc định (không tắt hay thêm gì), khớp với giả thuyết này; muốn chắc hơn thì chạy lại benchmark với model train có thêm blur/noise.
- **Lỗi mô phỏng có giống lỗi thật không?** Không hoàn toàn. Vết bẩn hay giọt nước thật thường chỉ che một phần ảnh. Nhiễu ban đêm thật phụ thuộc tín hiệu và đi kèm thiếu sáng, motion blur. Cháy sáng thật có lóa. Benchmark này là phép thử có kiểm soát, không thay được dữ liệu thật, nên bước tiếp theo là ảnh lỗi thật có nhãn.
- **Ngưỡng 0.70 và 0.85 lấy ở đâu?** Nhóm chọn trên chính 20 điều kiện này để minh họa, nên kết quả lạc quan. Trên xe thật cần hiệu chỉnh cho từng camera và kiểm chứng trên dữ liệu khác.
- **Vì sao mean conf ở blur σ = 5.5 vẫn 0.887?** Chỉ số này chỉ tính trên ảnh còn ít nhất 1 box. Số box/ảnh đã giảm từ 4.37 xuống 1.72 và recall từ 0.414 xuống 0.205. Confidence của các box còn sót không cho biết bao nhiêu vật bị bỏ sót.
- **Vì sao chọn YOLOv8n?** Model nhỏ: khoảng 3 triệu tham số, 8.1 GFLOPs, 3.5 ms/ảnh trên T4 (log notebook mục 8), hợp với phần cứng trên xe. Đổi lại, độ chính xác thấp: mAP@50 = 0.362 trên cả tập valid.

## Tự kiểm trước pitch

- [x] Có nền tảng, tính năng và sensor cụ thể (báo cáo mục 1).
- [x] Có metric định lượng, baseline và điều kiện lỗi (mục 3; baseline là mức 0 trên cùng 200 ảnh).
- [x] Có log/ảnh/plot và một failure case (notebook và thư mục `results/`).
- [x] Có link nguồn, commit/version, dataset, lệnh chạy (khối Truy vết ở mục 2, gồm cả `train.py`).
- [x] Phân biệt kết luận nguồn với kết quả nhóm tự đo (nhãn **[Paper]** và **[Nhóm đo]**).
- [x] Có một cải tiến hoặc fallback gắn với failure case (mục 5).
- [ ] Cả 5 thành viên có bản báo cáo riêng: mỗi người copy `step6_report.md` thành `step6_report_<tên>.md`, điền tên, tự viết lại 5 mục bằng lời của mình. Giữ nguyên số liệu, hình và link.
