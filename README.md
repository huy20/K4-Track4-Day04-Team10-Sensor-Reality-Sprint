# K4-Track4-Day04-Team10 — Sensor Reality Sprint

**Camera Degradation Health Score cho ADAS** — đo mức suy giảm của tính năng nhận diện (AEB/FCW) khi ảnh
camera bị nhòe, lệch phơi sáng và nhiễu cảm biến, đồng thời kiểm tra tương quan giữa các proxy metric
chất lượng ảnh và độ chính xác của detector.

> **Dataset**: [Roboflow ADAS `guna-tmuht/adas-oxrvp` v1](https://universe.roboflow.com/guna-tmuht/adas-oxrvp/dataset/1)
> · **Model**: [YOLOv8n (Ultralytics)](https://github.com/ultralytics/ultralytics)
> · **Chạy**: `notebooks/camera_degradation_benchmark.ipynb` (Kaggle GPU) hoặc `python train.py`

---

## 1. Bài toán

| Mục | Nội dung |
|---|---|
| Nền tảng | Xe ADAS / tự lái |
| Tính năng | Object detection (YOLOv8n) cho **AEB** (phanh khẩn cấp) & **FCW** (cảnh báo va chạm) |
| Sensor | Front monocular camera (camera quan sát phía trước) |
| Failure case | Defocus/Gaussian blur, motion blur, underexposure (đi đêm/chui hầm), overexposure (chói), sensor noise (ISO cao) |
| Dataset | Roboflow `guna-tmuht/adas-oxrvp` v1 — 15 class (`car`, `person`, `truck`, `traffic light`, `traffic sign`, ...) |
| Model | `yolov8n` (hoặc weights huấn luyện sẵn upload qua Kaggle Input) |

---

## 2. Claim ban đầu

> Khi ảnh bị nhòe mạnh hơn (sigma tăng) hoặc phơi sáng lệch khỏi mức chuẩn, `lap_var` giảm,
> `brightness` lệch khỏi 0.5, `entropy` giảm; và `mAP@50` cùng mean confidence của detector **giảm**
> theo mức suy giảm.

Claim là **giả thuyết để thử**, không phải kết luận. Kết quả thực tế xem mục 6 và [report.md](report.md).

---

## 3. Metric

| Nhóm | Metric | Đơn vị |
|---|---|---|
| Chất lượng ảnh (proxy) | `lap_var` (Laplacian variance) | variance |
| Chất lượng ảnh (proxy) | `brightness` (mean luminance) | 0..1 |
| Chất lượng ảnh (proxy) | `entropy` (Shannon) | bits |
| Chất lượng ảnh (proxy) | `health` (tổng hợp blur/exposure/entropy) | 0..1 |
| Detector | `mAP@50`, `mAP@50-95` | 0..1 |
| Detector | `precision`, `recall` | 0..1 |
| Detector | mean max confidence, số box/ảnh | 0..1, count |

---

## 4. Nguyên tắc so sánh

**Baseline** = ảnh val gốc (identity, level 0). **Degraded** = cùng ảnh đó sau khi áp một degradation ở
một mức. Label giữ nguyên, dùng **cùng hàm `model.val()`** cho mọi điều kiện. Chỉ ảnh đổi → chênh lệch
quy về chất lượng sensor, không phải cách đo.

> ⚠️ Mọi mức degradation chạy trên **200 ảnh val** (`MAX_VAL_IMAGES = 200`), nên baseline so sánh là dòng
> **identity trên 200 ảnh đó** (`mAP@50 = 0.466`). Dòng `clean` trong CSV chạy full valid (`0.362`) —
> dùng cho tham chiếu, không so trực tiếp. Chi tiết ở [report.md](report.md) mục 4.

---

## 5. Cấu trúc repo

```
.
├── README.md                                 # file này
├── report.md                                 # báo cáo kết quả
├── train.py                                  # script train YOLOv8 (baseline)
├── figure/                                   # kết quả chạy benchmark
│   ├── degradation_results.csv               # bảng số liệu đầy đủ
│   ├── curves.png                            # mAP / confidence / health vs mức suy giảm
│   ├── correlation.png                       # tương quan proxy ↔ detector
│   └── failure_cases.png                     # ảnh baseline vs degraded
└── notebooks/
    └── camera_degradation_benchmark.ipynb    # tải dataset, tạo degraded set, benchmark, plot
```

---

## 6. Kết quả chính (chi tiết ở [report.md](report.md))

Baseline identity (200 ảnh val) `mAP@50 = 0.466`.

| Lỗi | Mức nặng nhất | Δ mAP@50 |
|---|---|---|
| Sensor noise | sigma 35 | −81.5% (→ 0.086) |
| Motion blur | kernel 15 | −54.8% (→ 0.211) |
| Gaussian blur | sigma 5.5 | −52.1% (→ 0.223) |
| Underexposure | gamma 3.2 | −29.1% (→ 0.330) |
| Overexposure | gamma 0.4 | −9.0% (→ 0.424) |

- **Confidence bền giả tạo với blur**: mAP −52% nhưng mean confidence vẫn ~0.89 → không dùng confidence
  đơn thuần để tự chẩn đoán blur.
- **`lap_var` bị noise đánh lừa**: giảm khi blur (616 → 2) nhưng **tăng vọt** khi nhiễu (616 → 9475).
- **`mean_boxes` là chỉ báo detector tốt nhất** (r = 0.95 với mAP@50), trên cả `mean_conf` (r = 0.82).

---

## 7. Cách chạy

1. Mở notebook trên Kaggle.
2. Settings → **Internet: On**; **Accelerator: GPU T4**.
3. Add-ons → Secrets → thêm `ROBOFLOW_API_KEY` (không hard-code key vào notebook).
4. (Tuỳ chọn) Add Input → chọn Kaggle Model chứa weights `.pt` của nhóm.
5. Run All. Notebook tự dò weights trong `/kaggle/input/**/*.pt`; nếu không có thì train `yolov8n` trên bộ sạch.

Tham số chỉnh nhanh trong cell cấu hình: `EPOCHS`, `IMGSZ`, `BATCH`, `MAX_VAL_IMAGES`,
`SAMPLE_FOR_QUALITY`, `DEGRADATIONS`.

**Hoặc chạy train nhanh bằng script:**

```bash
pip install ultralytics roboflow opencv-python pandas matplotlib tqdm
python train.py    # sửa đường dẫn 'data' trong train.py cho khớp máy bạn
```

---

## 8. Ứng dụng cho ADAS (đề xuất)

Vì `lap_var` bị noise đánh lừa, hệ thống nên gate theo `mean_boxes` (số detection/ảnh) kết hợp `health`,
thay vì chỉ một "blur score":

1. **An toàn** (`mean_boxes ≥ 3.5`, `health ≥ 0.8`): ảnh đủ tốt, dùng camera.
2. **Cảnh báo** (`mean_boxes` 2–3.5 hoặc `health` 0.6–0.8): giảm tin cậy camera, ưu tiên fusion **Radar/LiDAR**.
3. **Nguy hiểm** (`mean_boxes < 2` hoặc `health < 0.6`): cảnh báo tài xế "camera bị mờ", **giảm tốc độ** giữ
   khoảng cách an toàn.

Ngưỡng minh hoạ từ sweep này (baseline `mean_boxes` = 4.37). Ngoài ra: thêm augmentation blur/noise vào
train, và thu dữ liệu camera đêm/thời tiết xấu thực tế để fine-tune.

---

## 9. Phân công (5 thành viên)

| Thành viên | Vai trò chính | Sản phẩm phụ trách |
|---|---|---|
| [Tên 1] | Tìm paper/repo, chốt claim & scope | Mục Tài liệu tham khảo, claim |
| [Tên 2] | Dataset + code degradation | Cell degradation, build degraded set |
| [Tên 3] | Train baseline / load weights | `best.pt`, `train.py`, log train |
| [Tên 4] | Chạy benchmark, metric, plot | `figure/degradation_results.csv`, `curves.png`, `correlation.png` |
| [Tên 5] | Failure case + tổng hợp báo cáo/slide | `figure/failure_cases.png`, `report.md`, slide |

> Nhiều người có thể cùng phụ trách một việc. Mỗi thành viên nộp một bản riêng trên VLearn.

---

## 10. Tài liệu tham khảo

- Roboflow Universe — ADAS dataset: https://universe.roboflow.com/guna-tmuht/adas-oxrvp/dataset/1
- Ultralytics YOLO (repo + docs): https://github.com/ultralytics/ultralytics
- Sakaridis et al., *ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving
  Scene Understanding*, ICCV 2021.
- Kenk & Hassaballah, *DAWN: Vehicle Detection in Adverse Weather Nature Dataset*, 2020.
- Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020.
- Loh & Chan, *Getting to Know Low-light Images with The Exclusively Dark Dataset*, CVIU 2019.
- Pech-Pacheco et al., *Diatom autofocusing in brightfield microscopy: a comparative study*, ICPR 2000
  (nguồn của focus measure bằng Laplacian variance).

---

## 11. Lưu ý

- Không commit API key. Dùng Kaggle Secret `ROBOFLOW_API_KEY`.
- Input Kaggle là read-only; cần ghi/convert thì copy sang `/kaggle/working` trước.
- Baseline trong `degradation_results.csv`: dòng `clean` chạy trên full valid (mAP@50 = 0.362), còn các
  mức degradation chạy trên 200 ảnh (baseline identity = 0.466). So sánh phải dùng cùng tập ảnh.
