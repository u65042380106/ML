import psycopg2
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import contextlib
import sys

def fetch_img_links_in_range(start_id, end_id):
    conn = psycopg2.connect(
        dbname="postgres",
        user="aitisi",
        password="tisi@2025",
        host="aitisi.tisi.lo",
        port="5432"
    )
    cur = conn.cursor()
    try:
        query = """
            SELECT id, img FROM tk5_aiget
            WHERE id BETWEEN %s AND %s
            ORDER BY id ASC
        """
        cur.execute(query, (start_id, end_id))
        rows = cur.fetchall()
        result = []
        for row in rows:
            id_ = row[0]
            img_links = row[1]
            if img_links and img_links.strip() != "":
                links_list = [link.strip() for link in img_links.split("\n") if link.strip()]
                result.append((id_, links_list))
            else:
                result.append((id_, []))
        return result
    finally:
        cur.close()
        conn.close()

def detect_tis_mark(yolo_model, img_url):
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # suppress stdout (YOLO prints to console)
        with contextlib.redirect_stdout(None):
            results = yolo_model(img)

        classes_detected = []
        for result in results:
            classes_detected += [result.names[int(cls)] for cls in result.boxes.cls]
        return "tis_mark" in classes_detected
    except Exception:
        # suppress error message entirely
        return False

def check_tis_mark_in_range(start_id, end_id):
    records = fetch_img_links_in_range(start_id, end_id)
    if not records:
        print(f"No records found between id {start_id} and {end_id}")
        return

    yolo_model = YOLO("runs/detect/tis_detect_model/weights/best.pt")

    for id_, links in records:
        if not links:
            print(f"id {id_}: No Image link")
            continue

        for link in links:
            if detect_tis_mark(yolo_model, link):
                print(f"id {id_}: YES")
                break
        else:
            print(f"id {id_}: NO")

if __name__ == "__main__":
    start_id = input("กรุณาใส่ id เริ่มต้น: ")
    end_id = input("กรุณาใส่ id สิ้นสุด: ")
    check_tis_mark_in_range(start_id, end_id)
