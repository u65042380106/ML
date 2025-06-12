import os
import random
from PIL import Image, ImageDraw
import numpy as np

# CONFIG
NUM_IMAGES = 50
IMG_SIZE = 640
LOGO_SIZE = 100
OUTPUT_DIR = "sim_dataset"
CLASS_ID = 0  # 0 = tis_mark

# เตรียมโฟลเดอร์
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

# โหลดโลโก้ มอก. 
logo = Image.open("tis_logo.png").convert("RGBA")
logo = logo.resize((LOGO_SIZE, LOGO_SIZE))

for i in range(NUM_IMAGES):
    # สร้างพื้นหลังสุ่มสี
    bg_color = tuple(np.random.randint(150, 255, 3))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=bg_color)
    
    # เลือกตำแหน่งสุ่มให้โลโก้
    max_x = IMG_SIZE - LOGO_SIZE
    max_y = IMG_SIZE - LOGO_SIZE
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # แปะโลโก้ลงพื้นหลัง
    img.paste(logo, (x, y), logo)

    # สร้าง label แบบ normalized
    xc = (x + LOGO_SIZE / 2) / IMG_SIZE
    yc = (y + LOGO_SIZE / 2) / IMG_SIZE
    w = LOGO_SIZE / IMG_SIZE
    h = LOGO_SIZE / IMG_SIZE

    # บันทึกภาพ
    filename = f"image_{i:03d}.jpg"
    img.save(f"{OUTPUT_DIR}/images/train/{filename}")

    # บันทึก label
    label_txt = f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
    with open(f"{OUTPUT_DIR}/labels/train/{filename.replace('.jpg', '.txt')}", "w") as f:
        f.write(label_txt)

print(" สร้าง dataset จำลองเสร็จแล้ว:", OUTPUT_DIR)
