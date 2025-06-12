from ultralytics import YOLO

# โหลดโมเดลเปล่า YOLOv8 (s, m, l)
model = YOLO("yolov8s.pt")

# ฝึกโมเดล
model.train(
    data="sim_dataset/data.yaml",  # ชี้ไปที่ไฟล์ data.yaml
    epochs=120,                     # จำนวนรอบการฝึก 
    imgsz=640,                     # ขนาดภาพ input
    batch=16,                      # จำนวนภาพต่อรอบการฝึก
    name="tis_detect_multi_model",       # ชื่อ project
)
