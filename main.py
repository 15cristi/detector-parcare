import csv
import cv2
import numpy as np
import threading
import os
import re
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from picamera2 import Picamera2
from ultralytics import YOLO
from sort import Sort
from util import get_car, is_valid_plate
from paddleocr import PaddleOCR
import requests
import lgpio
from gpiozero import DigitalInputDevice
import board
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from requests.auth import HTTPBasicAuth
from flask import Flask, jsonify
from flask_cors import CORS

# ==== Configurare motor pas cu pas (bariera intrare/iesire) ====
motor_pins = [14, 15, 18, 23]
sequence = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

def rotate_motor(h, pins, steps, delay=0.002, reverse=False):
    seq = list(reversed(sequence)) if reverse else sequence
    for step in range(steps):
        for half_step in seq:
            for i in range(4):
                lgpio.gpio_write(h, pins[i], half_step[i])
            time.sleep(delay)

def show_barrier_message(locuri_libere, message):
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "Sistem Parcare", font=font, fill=255)
    draw.text((0, 20), message, font=font, fill=255)
    draw.text((0, 40), f"Locuri libere: {locuri_libere}", font=font, fill=255)
    oled.image(image)
    oled.show()

def deschide_bariera():
    h = lgpio.gpiochip_open(0)
    for pin in motor_pins:
        lgpio.gpio_claim_output(h, pin)

    try:
        libere = locuri_libere()
        show_barrier_message(libere, "Ridic bariera")
        print("[INFO] 🔐 Ridic bariera")
        rotate_motor(h, motor_pins, 170)
        time.sleep(5)
        show_barrier_message(libere, "Cobor bariera")
        print("[INFO] 🔑 Cobor bariera")
        rotate_motor(h, motor_pins, 170, reverse=True)
    finally:
        for pin in motor_pins:
            lgpio.gpio_write(h, pin, 0)
        lgpio.gpiochip_close(h)
        print("[INFO] ✅ Bariera resetata")

def trimite_access_log(plate_number):
    url = "http://192.168.1.131:8080/api/access"
    auth = HTTPBasicAuth('admin1', 'test')
    payload = {"licensePlate": plate_number}
    try:
        response = requests.post(url, json=payload, auth=auth)
        if response.status_code == 200:
            print(f"[INFO] ? Log trimis pentru {plate_number}")
        else:
            print(f"[ERROR] Logul NU a fost trimis (status {response.status_code})")
    except Exception as e:
        print(f"[EROARE] Conexiune la /api/access: {e}")

# ==== Senzori ==== 
sensor1 = DigitalInputDevice(17)
sensor2 = DigitalInputDevice(27)
sensor_exit = DigitalInputDevice(16)

pause_detection_event = threading.Event()
pause_detection_event.clear()

last_exit_trigger_time = 0
EXIT_DELAY = 5


def locuri_libere():
    return int(sensor1.value) + int(sensor2.value)

# ==== OLED SSD1306 ====
i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)
oled.fill(0)
oled.show()
font = ImageFont.load_default()

def update_display(locuri):
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "Sistem Parcare", font=font, fill=255)
    draw.text((0, 20), "Locuri libere:", font=font, fill=255)
    draw.text((0, 40), f"{locuri}", font=font, fill=255)
    oled.image(image)
    oled.show()


# ==== Camera, modele YOLO, OCR ====
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.start()

coco_model = YOLO('yolov8m.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
mot_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.5)
ocr_model = PaddleOCR(use_angle_cls=True, lang='latin', det_db_box_thresh=0.6,
                      rec_algorithm='CRNN', use_gpu=False)

vehicles = [2]
results = {}
frame_nmr = 0
running = True
processed_cars = set()
ocr_queue = Queue(maxsize=30)
executor = ThreadPoolExecutor(max_workers=3)

os.makedirs("plates", exist_ok=True)

def verifica_baza_date(plate_number):
    url = f"http://192.168.1.131:8080/api/vehicles/{plate_number}"
    auth = HTTPBasicAuth('admin1', 'test')
    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            print(f"[INFO] Numarul {plate_number} este autorizat.")
            return True
        elif response.status_code == 404:
            print(f"[INFO] Numarul {plate_number} nu este gasit in baza de date.")
            return False
        else:
            print(f"[EROARE] Eroare la verificare: {response.status_code}")
            return False
    except Exception as e:
        print(f"[EROARE] Conexiune la server: {e}")
        return False
def is_valid_romanian_plate(plate):
    return re.match(r'^[A-Z]{1,2}[0-9]{2,3}[A-Z]{3}$', plate) is not None


def process_ocr(frame_nmr, car_id, license_plate_crop):
    plate_filename = f"plates/crop_{frame_nmr}_{car_id}.jpg"
    cv2.imwrite(plate_filename, license_plate_crop)

    try:
        gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = ocr_model.ocr(binary, cls=True)
        print("[DEBUG] OCR:", result)

        if result and len(result[0]) > 0 and len(result[0][0]) >= 2:
            text, score = result[0][0][1]
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper().replace(" ", "").replace("-", "").strip())

            print(f"[INFO] Placuta curatata: {clean_text}")

            if clean_text and is_valid_romanian_plate(clean_text) and verifica_baza_date(clean_text):
                print(f"[INFO] ? {clean_text} autorizat - deschid bariera")
                trimite_access_log(clean_text)
                pause_detection_event.set()
                time.sleep(2)
                deschide_bariera()
                processed_cars.add(car_id)
                time.sleep(3)
                pause_detection_event.clear()

                # Asteapta 5 secunde si apoi sterge masina din set
                time.sleep(5)
                try:
                    processed_cars.remove(car_id)
                    print(f"[INFO] {clean_text} - car_id {car_id} resetat din processed_cars.")
                except KeyError:
                    print(f"[WARN] car_id {car_id} nu a fost gasit in processed_cars pentru eliminare.")
            else:
                print(f"[INFO] {clean_text} nu este autorizat.")
        else:
            print("[INFO] Nu s-a detectat text valid.")
    finally:
        if os.path.exists(plate_filename):
            os.remove(plate_filename)

last_parking_status = {"slot1": -1, "slot2": -1}


def process_detection():
    global frame_nmr, running, last_parking_status

    while running:
        if pause_detection_event.is_set():
            time.sleep(0.5)
            continue

        libere = locuri_libere()
        update_display(libere)

        current_status = {
            "slot1": int(not sensor1.value),
            "slot2": int(not sensor2.value)
        }

        if current_status != last_parking_status:
            print(f"[INFO] ?? Status locuri schimbat: {current_status}")
            last_parking_status = current_status.copy()

            try:
                requests.post("http://192.168.1.131:8080/api/slot-status", json=current_status)
            except Exception as e:
                print(f"[EROARE] Nu am putut trimite status: {e}")

        if libere == 0:
            print("[INFO] Niciun loc liber. Asteapta...")
            time.sleep(1)
            continue

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
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if not ocr_queue.full():
                        ocr_queue.put((frame_nmr, car_id, license_plate_crop))

        while not ocr_queue.empty():
            executor.submit(process_ocr, *ocr_queue.get())

def monitor_iesire():
    global running

    bariera_deschisa = False

    while running:
        # Senzorul tau e ACTIV cand valoarea e 0
        if sensor_exit.value == 0 and not bariera_deschisa:
            print("[IESIRE] Detectie la iesire. Ridic bariera...")
            pause_detection_event.set()
            deschide_bariera()
            pause_detection_event.clear()
            bariera_deschisa = True

            # Asteapta eliberarea senzorului
            while sensor_exit.value == 0 and running:
                time.sleep(0.1)

            print("[IESIRE] Senzor eliberat. Pregatit pentru urmatoarea ma?ina.")
            bariera_deschisa = False

        time.sleep(0.1)

# ==== Start Threads ====
detection_thread = threading.Thread(target=process_detection, daemon=True)
exit_thread = threading.Thread(target=monitor_iesire, daemon=True)

detection_thread.start()
exit_thread.start()

cv2.namedWindow("Camera Live", cv2.WINDOW_NORMAL)
while running:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False






cv2.destroyAllWindows()
picam2.stop()
running = False
detection_thread.join()
exit_thread.join()
executor.shutdown()
