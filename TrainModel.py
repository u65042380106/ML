from ultralytics import YOLO

# โหลดโมเดลเปล่า YOLOv8 (เลือกขนาดตามต้องการ: s, m, l)
model = YOLO("yolov8s.pt")

# ฝึกโมเดล
model.train(
    data="sim_dataset/data.yaml",  # ชี้ไปที่ไฟล์ data.yaml
    epochs=50,                     # จำนวนรอบการฝึก (เพิ่มหาก dataset เยอะ)
    imgsz=640,                     # ขนาดภาพ input
    batch=16,                      # จำนวนภาพต่อรอบการฝึก
    name="tis_detect_model",       # ชื่อ project
)
