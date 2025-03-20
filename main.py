from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from sort import Sort
from util import get_car, read_license_plate, write_csv, is_valid_plate

# Initializare camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (1000, 1000)})
picam2.configure(camera_config)
picam2.start()

# Initializare modele YOLO
print("[INFO] Incarcare modele YOLO...")
coco_model = YOLO('yolov8m.pt')  # Model pentru detectarea masinilor
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # Model pentru placute

# Initializare tracker SORT
mot_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.5)

# Detectam doar masini (clasa 2)
vehicles = [2]

results = {}
frame_nmr = 0
running = True

# Creare thread separat pentru detectie (YOLO si OCR)
def process_detection():
    global frame_nmr, results, running

    while running:
        frame_nmr += 1
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results[frame_nmr] = {}

        # Detectare masini cu YOLO
        detections = coco_model(frame, imgsz=640, conf=0.3, iou=0.5)[0]
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Car {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tracking SORT
        if detections_:
            track_ids = mot_tracker.update(np.array(detections_))

            # Detectare placute (DOAR pentru masinile detectate)
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Convertim la grayscale
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                    # Crestem contrastul
                    license_plate_crop_gray = cv2.equalizeHist(license_plate_crop_gray)

                    # Thresholding Otsu
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )

                    # OCR pe un thread separat (pentru viteza maxima)
                    threading.Thread(target=process_ocr, args=(frame_nmr, car_id, license_plate_crop_thresh)).start()

        write_csv(results, './test.csv')

def process_ocr(frame_nmr, car_id, license_plate_crop_thresh):
    """Functie care ruleaza OCR pe un thread separat pentru performanta mai buna."""
    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

    if license_plate_text and is_valid_plate(license_plate_text) and license_plate_text_score > 0.5:
        results[frame_nmr][car_id] = {'license_plate': {'text': license_plate_text}}
        print(f"[INFO] Numar detectat: {license_plate_text}")

# Creare un thread separat pentru detectie
detection_thread = threading.Thread(target=process_detection, daemon=True)
detection_thread.start()

# Fereastra principala - Camera Live
cv2.namedWindow("Camera Live", cv2.WINDOW_NORMAL)

while running:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Afisam fluxul video continuu
    cv2.imshow("Camera Live", frame)

    # Daca apasam 'q', iesim
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Oprire si curatare resurse
cv2.destroyAllWindows()
picam2.stop()
running = False
detection_thread.join()
