# Camera Degradation Health Score Benchmark cho ADAS (Autonomous Driving)
**Nhóm 10 — K4 Track4 Day04 Sensor Reality Sprint**

> **Commit Version**: `73d1882` (Merge pull request #1 from huy20/train)  
> **Dataset**: [Roboflow ADAS Dataset `guna-tmuht/adas-oxrvp` v1](https://universe.roboflow.com/guna-tmuht/adas-oxrvp/dataset/1)  
> **Pre-trained / Model Base**: [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)  
> **Lệnh chạy Benchmark**: `python train.py` hoặc chạy toàn bộ file notebook `notebooks/camera_degradation_benchmark.ipynb`

---

## 📌 BÁO CÁO NHÓM (DÙNG ĐỂ PITCH 3-5 PHÚT & NỘP BÀI)

### 1. Problem (Vấn đề & Nền tảng)
- **Nền tảng & Tính năng**: Hệ thống ADAS/Tự lái nâng cao trang bị tính năng **Tự động phanh khẩn cấp (AEB)** và **Cảnh báo va chạm phía trước (FCW)**.
- **Sensor**: Cảm biến **Front Monocular Camera** (Camera quan sát phía trước xe).
- **Failure thực tế**: Trong điều kiện vận hành thực tế, camera thường xuyên gặp sự cố mờ kính gió (defocus blur), nhòe do va sóc / chuyển động nhanh (motion blur), phơi sáng kém khi chui hầm / đi đêm (underexposure), chói sáng sáng sớm (overexposure) hoặc nhiễu hạt ISO cao (sensor noise).
- **Hậu quả**: Các lỗi suy giảm chất lượng ảnh mức pixel làm giảm confidence của mô hình 2D Perception (YOLOv8), dẫn đến bỏ sót vật thể (False Negative) gây nguy cơ va chạm nguy hiểm.

---

### 2. Method (Phương pháp & Thuật toán)
- **Thuật toán / Model**: YOLOv8n (Object Detection 2D).
  - **Input**: Ảnh RGB kích thước \(640 \times 640 \times 3\).
  - **Output**: Bounding boxes \((x, y, w, h)\), Object confidence score \([0, 1]\), Class ID (15 classes: Car, Person, Bicycle, Bus, Motorcycle, Traffic Light, Sign...).
  - **Metrics**: 
    - *Proxy Quality Metric*: Laplacian Variance (`lap_var`), Shannon Entropy, Brightness.
    - *Health Score*: Hàm chuẩn hóa tổng hợp chất lượng ảnh \(H \in [0, 1]\).
    - *Detector Metric*: mAP@50, mAP@50-95, Mean Max Confidence.
- **Giả định**: Ground-truth labels không đổi giữa tập ảnh gốc (Clean) và tập ảnh bị làm lỗi (Degraded). Thay đổi về mAP hoàn toàn do sự suy giảm của cảm biến camera.

---

### 3. Benchmark & Số đo Định lượng (Results)

*Dữ liệu tổng hợp từ thực nghiệm chạy trên Kaggle GPU T4 với 200 ảnh Validation của tập dataset Roboflow ADAS (`guna-tmuht/adas-oxrvp`).*

| Điều kiện / Lỗi | Mức độ suy giảm | Đô mờ (lap_var) | Accuracy (mAP50) | Đánh giá thực tế |
| :--- | :--- | :---: | :---: | :--- |
| **Ảnh sạch (Baseline)** | Chuẩn | **315.4** | **68.5%** | Xe chạy điều kiện lý tưởng |
| **Gaussian Blur** | Mờ nhẹ (Sigma 1.0) | 142.1 | 61.2% | Kính xe hơi mờ / đọng nước |
| **Gaussian Blur** | Mờ vừa (Sigma 3.5) | 38.6 | 38.5% | Mất nét rõ rệt, mAP giảm mạnh |
| **Gaussian Blur** | **Mờ nặng (Sigma 5.5)** | **12.3** | **16.2%** | **Điểm gãy: Mô hình không hoạt động** |
| **Motion Blur** | Rung vừa (Kernel 7) | 68.2 | 44.1% | Xe di chuyển nhanh |
| **Motion Blur** | **Rung nặng (Kernel 15)** | **22.5** | **20.1%** | Rung lắc cực mạnh khi qua gờ |
| **Thiếu sáng (Dark)** | Gamma 2.5 | 85.4 | 41.2% | Xe đi ca tối / chui hầm |
| **Nhiễu cảm biến (Noise)**| Noise Sigma 35 | 890.1* | 24.5% | *Nhiễu hạt làm sai số lap_var |

> **Điểm chính cần nhớ khi thuyết trình**:
> - **Khi ảnh sạch**: YOLOv8 đạt độ chính xác **68.5%**.
> - **Ngưỡng báo động**: Khi độ mờ `lap_var` rơi xuống dưới **40.0** (ví dụ mờ Sigma 3.5 trở lên), mAP giảm dốc đứng xuống còn dưới **38.5%** (giảm gần 50% hiệu năng).

---

### 4. Phân tích Tình huống Thất bại (Failure Case)
- **Trường hợp phân tích**: Ảnh bị rung mờ nặng (Motion Blur Kernel 15) khi xe qua gờ giảm tốc.
- **Hiện tượng**: 
  - Độ mờ `lap_var` sụt giảm từ **315.4** xuống chỉ còn **22.5**.
  - Mô hình YOLOv8 bỏ sót 01 **Người đi bộ** sang đường ở khoảng cách 18m.
- **Nguyên nhân**: Nhòe chuyển động làm mất vệt viền sắc nét của vật thể. Các bộ lọc tầng đầu của AI không nhận diện được viền, dẫn đến các tầng sau không đủ tín hiệu để nhận diện người.

---

### 5. Đề xuất Kỹ thuật cho ADAS (Engineering Decision)
Dựa trên số đo `lap_var`, hệ thống ADAS sẽ xử lý theo 3 mức:

1. **Vùng An toàn (`lap_var >= 100`)**: Ảnh đủ nét. Sử dụng 100% kết quả từ Camera.
2. **Vùng Cảnh báo (`40 <= lap_var < 100`)**: Ảnh bắt đầu mờ. Giảm độ tin tưởng vào Camera, chuyển sang ưu tiên dữ liệu từ **Radar / LiDAR** (Sensor Fusion).
3. **Vùng Nguy hiểm (`lap_var < 40`)**: Camera mất tác dụng. Phát cảnh báo cho tài xế *"Camera bị mờ - Vui lòng lái xe cẩn thận"*, đồng thời **tự động giảm tốc độ xe** để giữ khoảng cách an toàn lớn hơn.
3. **Dữ liệu & Cải tiến tiếp theo**:
   - Thêm các kiểu Augmentation mờ/nhiễu vào quá trình huấn luyện (Retrain YOLOv8 với Adverse Weather Augmentation).
   - Thu thập thêm dữ liệu camera đêm và thời tiết xấu thực tế tại Việt Nam để fine-tune.

---

## 📂 HƯỚNG DẪN TRUY VẾT & CHẠY LẠI (REPRODUCIBILITY)

```bash
# 1. Clone repository
git clone https://github.com/huy20/K4-Track4-Day04-Team10-Sensor-Reality-Sprint.git
cd K4-Track4-Day04-Team10-Sensor-Reality-Sprint

# 2. Cài đặt thư viện cần thiết
pip install ultralytics roboflow opencv-python pandas matplotlib tqdm

# 3. Chạy file train/benchmark
python train.py

# 4. Hoặc mở và chạy toàn bộ Jupyter Notebook:
# notebooks/camera_degradation_benchmark.ipynb
```
