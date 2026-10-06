# Camera suy giảm và detector ADAS: health score báo trước được tới đâu?

**Nhóm 10 · K4 Track4 Day04 Sensor Reality Sprint · Bước 6**  
**Người thực hiện:** Lê Trung Kiên (MSSV: 02748)  
**Ngày nộp:** 06/10/2026  

> Ký hiệu: **[Paper]** = kết luận của paper/repo nguồn · **[Nhóm đo]** = kết quả nhóm tự chạy trong notebook.

---

## 1. Problem

- **Nền tảng:** Ô tô có trang bị hệ thống hỗ trợ lái xe nâng cao (ADAS), sử dụng 01 camera RGB nhìn phía trước.
- **Tính năng:** Phát hiện vật thể 15 lớp (xe ô tô, xe máy, người đi bộ, người lái xe, biển báo, đèn giao thông…) làm đầu vào quan trọng cho tính năng Cảnh báo va chạm phía trước (FCW) và Phanh khẩn cấp tự động (AEB).
- **Sensor và failure thực tế:** 
  - Ống kính bẩn, đọng nước mưa hoặc bị lệch nét $\rightarrow$ **Gaussian Blur**.
  - Xe rung lắc mạnh khi đi qua gờ giảm tốc với tốc độ cao $\rightarrow$ **Motion Blur**.
  - Xe đi vào hầm đường bộ hoặc di chuyển ca tối $\rightarrow$ **Underexposure** (Thiếu sáng).
  - Nắng gắt sáng sớm chiếu thẳng camera $\rightarrow$ **Overexposure** (Cháy sáng).
  - Cảm biến đẩy ISO cao khi thiếu sáng $\rightarrow$ **Sensor Noise** (Nhiễu hạt).
- **Rủi ro:** Mô hình AI detector sụt giảm độ chính xác và bỏ sót vật thể nghiêm trọng nhưng hệ thống không đưa ra bất kỳ cảnh báo lỗi nào (Silent False Negative).
- **Câu hỏi nghiên cứu:** Mức độ suy giảm hiệu năng của detector ứng với từng loại lỗi là bao nhiêu? Liệu một chỉ số chất lượng ảnh không nhãn (Health Score) có thể báo trước được nguy cơ này hay không?

---

## 2. Method

| Thành phần | Nội dung |
|---|---|
| **Detector** | YOLOv8n (Ultralytics 8.4.173), nhóm fine-tune trên dataset ADAS. **Input:** Ảnh RGB 640×640. **Output:** Bounding Box, Class ID (15 lớp), Confidence score. |
| **Cách đo** | Theo Michaelis et al. 2019: Giữ nguyên ảnh gốc và nhãn Ground-truth, chỉ làm hỏng ảnh theo 5 loại lỗi × 4 mức độ severity (cộng mức 0 baseline). |
| **Health score (0..1)** | Trung bình 3 chỉ số trên ảnh xám: <br>1. **Độ nét**: Laplacian Variance (`lap_var`) chuẩn hóa.<br>2. **Phơi sáng**: Hàm Gauss theo Mean Brightness và tỉ lệ pixel quá tối/quá sáng (`clip_low`/`clip_high`).<br>3. **Entropy**: Shannon Entropy chuẩn hóa. (Đo hoàn toàn bằng OpenCV trên CPU, không tốn GPU). |
| **Giả định** | Các lỗi mô phỏng pixel đại diện cho lỗi cảm biến thực tế; vị trí vật thể không đổi nên giữ nguyên nhãn gốc; 200 ảnh validation đại diện cho phân bố dữ liệu. |

**[Paper]** Michaelis et al. 2019 (*Benchmarking Robustness in Object Detection*): Các mô hình detector tiêu chuẩn sụt giảm từ 30% đến 60% hiệu năng gốc khi hình ảnh đầu vào bị suy giảm chất lượng.

**Thông tin truy vết (Reproducibility)**:
- **Code & Notebook:** [`notebooks/camera_degradation_benchmark.ipynb`](https://github.com/huy20/K4-Track4-Day04-Team10-Sensor-Reality-Sprint/blob/main/notebooks/camera_degradation_benchmark.ipynb) (Commit `5684630`).
- **Dataset:** Roboflow [`guna-tmuht/adas-oxrvp` v1](https://universe.roboflow.com/guna-tmuht/adas-oxrvp/dataset/1) (Train/Valid/Test: 6927/1980/987 ảnh).
- **Weights:** File `best (1).pt` (Fine-tune YOLOv8n 10 epochs).
- **Lệnh chạy tạo Figure/Bảng:** `python scripts/make_step6_figures.py`
- **Lệnh chạy tạo PDF Báo cáo:** `python scripts/build_report_pdf.py`

---

## 3. Benchmark [Nhóm đo]

- **Baseline:** Mức 0 (Clean) trên đúng 200 ảnh valid đầu tiên $\rightarrow$ **mAP@50 = 0.466**.
- **Metric đo lường:** mAP@50 (chính), Recall, Số box trung bình/ảnh (`mean_boxes`), Confidence trung bình (`mean_conf`), Health Score (`health`).

![Curves](../results/main_figure.png)

### Bảng Kết quả Benchmark (Tóm tắt mức nặng nhất từng loại lỗi):

| Loại Suy giảm (Degradation) | Mức lỗi (Severity) | mAP@50 | Biến thiên Δ mAP@50 | Recall | Box/ảnh | Mean Conf | Health Score | Đánh giá |
|---|---|:---:|:---:|:---: |:---:|:---:|:---:|---|
| **Baseline (Ảnh gốc)** | **Mức 0** | **0.466** | **—** | **0.414** | **4.37** | **0.941** | **0.931** | Chuẩn |
| **Gaussian Blur** | Sigma = 5.5 px | 0.223 | **−52.1%** | 0.205 | 1.72 | 0.887 | 0.607 | Mờ nét nặng |
| **Motion Blur** | Kernel = 15 px | 0.211 | **−54.8%** | 0.191 | 2.13 | 0.838 | 0.632 | Xe rung lắc mạnh |
| **Underexposure** | Gamma = 3.2 | 0.330 | **−29.1%** | 0.288 | 3.36 | 0.942 | 0.579 | Thiếu sáng nặng |
| **Overexposure** | Gamma = 0.4 | 0.424 | **−9.0%** | 0.415 | 4.18 | 0.937 | 0.920 | Cháy sáng nhẹ |
| **Sensor Noise** | Sigma = 35 | **0.086** | **−81.5%** | **0.067** | **0.24** | **0.619** | **0.939** | **Mô hình mù hoàn toàn** |

> **Phân tích chi tiết của Nhóm:**
> 1. **Noise là nguy hiểm nhất:** Nhiễu cảm biến làm mAP@50 sụt giảm khủng hại **−81.5%** (từ 0.466 xuống 0.086).
> 2. **Confidence bị đánh lừa khi Blur:** Khi mAP sụt giảm tới 52.1% do Blur, chỉ số `mean_conf` vẫn duy trì ở mức rất cao (**0.887**) $\rightarrow$ Detector vẫn "tự tin giả tạo" trên các box ít ỏi còn sót lại.
> 3. **Bẫy Laplacian Variance đối với Noise:** Khi bị nhiễu hạt, `lap_var` không những không giảm mà còn **tăng vọt từ 616 lên 9475** (do nhiễu tạo ra các chi tiết sắc nét giả), dẫn đến `Health Score` đọc nhầm thành ảnh rất nét (0.939).

---

## 4. Failure Case Analysis (Gaussian Blur $\sigma = 5.5$ px)

![Failure Case](../results/failure_case_blur_sigma5p5_compact.png)

- **Hiện tượng [Nhóm đo]:** Trong ví dụ trên, ở ảnh gốc baseline, YOLOv8 phát hiện chính xác Xe tải (0.91), Xe máy (0.93) và Người lái xe (0.92). Khi bị mờ ($\sigma = 5.5$), toàn bộ các Bounding Box **hoàn toàn biến mất (Confidence rơi về 0.00)**.
- **Nguyên nhân kỹ thuật:** Hiệu ứng Blur làm triệt tiêu các dải tần số cao (mất cạnh biên/edge gradient). Các lớp Convolutional đầu tiên của YOLOv8 không bắt được nét biên, dẫn đến Feature Map ở các lớp sâu không đủ tín hiệu kích hoạt NMS.
- **Tác động ADAS:** Tính năng FCW/AEB bị vô hiệu hóa hoàn toàn im lặng, gây ra va chạm nguy hiểm trực tiếp.

---

## 5. Engineering Decision & Fallback Strategy

### A. Quy tắc Cảnh báo và Fallback (Fail-Safe Matrix)
Để khắc phục nhược điểm Health Score bị qua mặt bởi Noise, nhóm đề xuất **Luật kết hợp 2 điều kiện**:
$$\text{Kích hoạt Fallback KHIS: } (\text{Health Score} < 0.70) \quad \mathbf{HO\breve{A}C} \quad (\text{Mean Confidence} < 0.85)$$

| Luật kiểm tra | Số case phát hiện (/9 case suy giảm nặng) | Số case báo sai (/11 case nhẹ) | Kết luận |
|---|:---:|:---:|---|
| Chỉ dùng `Health < 0.70` | 6/9 | 3/11 | Bỏ sót 3 mức Noise (do Noise làm tăng lap_var) |
| Chỉ dùng `Mean Conf < 0.85` | 4/9 | 0/11 | Bỏ sót các ca Blur nặng (do Conf giữ cao giả tạo) |
| **Kết hợp cả 2 Luật** | **9/9 (100%)** | **3/11** | **Bắt trọn 100% các tình huống suy giảm nguy hiểm** |

### B. Hành động hệ thống khi kích hoạt Fallback:
1. Gửi cảnh báo hình ảnh/âm thanh cho tài xế: *"Camera suy giảm - Vui lòng kiểm soát tay lái"*.
2. Giảm độ tin tưởng vào Camera, chuyển sang ưu tiên cảm biến **LiDAR / Radar** (Dynamic Sensor Fusion).
3. Chủ động **hạ ODD**: Giảm tốc độ tối đa của xe và tăng khoảng cách an toàn với xe phía trước.

---

## 6. Phân công đóng góp cá nhân (Lê Trung Kiên - 02748)
- Thực hiện kéo/đồng bộ và kiểm chứng code repository (`report.md`, `make_step6_figures.py`, `build_report_pdf.py`).
- Đánh giá và tổng hợp bảng kiểm tra luật Fallback (`results/fallback_rule_check.csv`).
- Hoàn thiện báo cáo cá nhân chuẩn Bước 6 nộp hệ thống VLearn.
