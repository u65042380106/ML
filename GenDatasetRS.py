import os
import random
from PIL import Image
import numpy as np

# CONFIG
NUM_IMAGES = 120
IMG_SIZE = 640
LOGO_MIN_SIZE = 50
LOGO_MAX_SIZE = 150
OUTPUT_DIR = "sim_dataset_multi"
# กำหนดโลโก้และ class_id
logos = [
    {"path": "tis_logo.png", "class_id": 0},
    {"path": "tis_logo2.png", "class_id": 1},
    {"path": "tis_logo3.png", "class_id": 2},
]

# เตรียมโฟลเดอร์
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

# โหลดโลโก้แต่ละแบบ
for logo in logos:
    logo["img"] = Image.open(logo["path"]).convert("RGBA")

for i in range(NUM_IMAGES):
    bg_color = tuple(np.random.randint(150, 255, 3))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=bg_color)

    num_objects = random.randint(1, len(logos))  # จำนวนโลโก้ในภาพ 1 - จำนวนโลโก้ทั้งหมด

    labels = []
    occupied_areas = []  # เก็บพื้นที่ที่วางโลโก้ไว้ เพื่อหลีกเลี่ยงซ้อนทับ (ถ้าต้องการ)

    for _ in range(num_objects):
        logo_info = random.choice(logos)

        logo_size = random.randint(LOGO_MIN_SIZE, LOGO_MAX_SIZE)
        logo_resized = logo_info["img"].resize((logo_size, logo_size), Image.Resampling.LANCZOS)


        # หา position ที่ไม่ซ้อนกับโลโก้อื่น 
        max_x = IMG_SIZE - logo_size
        max_y = IMG_SIZE - logo_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # แปะโลโก้
        img.paste(logo_resized, (x, y), logo_resized)

        # สร้าง label แบบ normalized
        xc = (x + logo_size / 2) / IMG_SIZE
        yc = (y + logo_size / 2) / IMG_SIZE
        w = logo_size / IMG_SIZE
        h = logo_size / IMG_SIZE

        label_txt = f"{logo_info['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        labels.append(label_txt)

    # บันทึกภาพ
    filename = f"image_{i:03d}.jpg"
    img.save(f"{OUTPUT_DIR}/images/train/{filename}")

    # บันทึก label
    with open(f"{OUTPUT_DIR}/labels/train/{filename.replace('.jpg', '.txt')}", "w") as f:
        f.write("\n".join(labels) + "\n")

print("สร้าง dataset หลายคลาสเสร็จแล้ว:", OUTPUT_DIR)
