import os
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกไว้
model = YOLO("runs/detect/tis_detect_model/weights/best.pt")

# โฟลเดอร์ที่มีภาพทดสอบ
image_folder = "C:\\PRR\\dd\\data"

# วนลูปตรวจแต่ละภาพ
for fname in os.listdir(image_folder):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_folder, fname)
        results = model.predict(image_path, conf=0.8)
        
        # เช็กจำนวน object ที่เจอ
        num_objects = len(results[0].boxes)
        
        # แสดงผล
        if num_objects > 0:
            print(f"{fname}: ✅ มีตรา มอก.")
        else:
            print(f"{fname}: ❌ ไม่มีตรา มอก.")
