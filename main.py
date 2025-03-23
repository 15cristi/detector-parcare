import csv
import cv2
import numpy as np
import threading
import os
import re
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from picamera2 import Picamera2
from ultralytics import YOLO
from sort import Sort
from util import get_car, is_valid_plate
from paddleocr import PaddleOCR
import requests
# Initializare camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Modele YOLO
coco_model = YOLO('yolov8m.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Tracker
mot_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.5)

# OCR
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.6, rec_algorithm='CRNN', rec_model_dir=None, rec_char_dict_path=None, rec_image_shape='3,32,100', rec_model='en_PP-OCRv4_rec_infer', det_model='en_PP-OCRv4_det_infer', cls_model='ch_ppocr_mobile_v2.0_cls_infer', use_gpu=False)


vehicles = [2]
results = {}
frame_nmr = 0
running = True
processed_cars = set()
ocr_queue = Queue(maxsize=30)
executor = ThreadPoolExecutor(max_workers=3)

with open('./test.csv', 'w', newline='') as f:
    csv.writer(f).writerow(["license_number"])

def process_ocr(frame_nmr, car_id, license_plate_crop):
    plate_filename = f"plates/crop_{frame_nmr}_{car_id}.jpg"
    os.makedirs("plates", exist_ok=True)
    cv2.imwrite(plate_filename, license_plate_crop)

    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = ocr_model.ocr(binary, cls=True)
    print("[DEBUG] PaddleOCR detections (preprocesat):", result)

    if result and len(result[0]) > 0 and len(result[0][0]) >= 2:
        text, score = result[0][0][1]
        print(f"[DEBUG] Text original detectat: '{text}' cu scor {score}")

        clean_text = text.upper().replace(" ", "").replace("-", "").strip()
        clean_text = re.sub(r'[^A-Z0-9]', '', clean_text)

        print(f"[DEBUG] Text curatat pentru scriere in CSV: '{clean_text}'")

        if clean_text:
            with open('./test.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([clean_text])
                f.flush()
            print(f"[INFO] Scris direct in CSV: {clean_text}")

            # ?terge imaginea dupa ce a fost scrisa in CSV
            try:
                os.remove(plate_filename)
                print(f"[INFO] Imaginea '{plate_filename}' a fost ?tearsa.")
            except Exception as e:
                print(f"[ERROR] Nu am putut sterge imaginea '{plate_filename}': {e}")
                os.remove(plate_filename)
        else:
            print("[WARNING] Textul curatat este gol dupa preprocesare.")
            os.remove(plate_filename)
    else:
        print("[WARNING] PaddleOCR a intors rezultat invalid sau gol!")



def process_detection():
    global frame_nmr, running

    while running:
        frame_nmr += 1
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detections = coco_model(frame, imgsz=640, conf=0.3, iou=0.5)[0]
        detections_ = [det[:5] for det in detections.boxes.data.tolist() if int(det[5]) in vehicles]

        if detections_:
            track_ids = mot_tracker.update(np.array(detections_))
            license_plates = license_plate_detector(frame, conf=0.2, iou=0.4)[0]

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, _, _ = license_plate
                _, _, _, _, car_id = get_car(license_plate, track_ids)

                if car_id != -1 and car_id not in processed_cars:
                    processed_cars.add(car_id)
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if not ocr_queue.full():
                        ocr_queue.put((frame_nmr, car_id, license_plate_crop))

        while not ocr_queue.empty():
            executor.submit(process_ocr, *ocr_queue.get())

# Start detection thread
detection_thread = threading.Thread(target=process_detection, daemon=True)
detection_thread.start()

# Main live feed loop
cv2.namedWindow("Camera Live", cv2.WINDOW_NORMAL)
while running:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
running = False
detection_thread.join()
executor.shutdown()