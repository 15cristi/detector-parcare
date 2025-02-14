from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import csv
from sort import Sort  # Algoritmul de tracking

# Functii auxiliare (get_car, write_csv, read_license_plate)
def get_car(license_plate_bbox, vehicle_bboxes):
    if len(license_plate_bbox) >= 4:
        x_lp, y_lp, x2_lp, y2_lp = license_plate_bbox[:4]
    else:
        print("Unexpected license_plate_bbox structure:", license_plate_bbox)
        return -1, -1, -1, -1, -1

    for vehicle_bbox in vehicle_bboxes:
        x_v, y_v, x2_v, y2_v, vehicle_id = vehicle_bbox
        if (x_lp > x_v and y_lp > y_v and x2_lp < x2_v and y2_lp < y2_v):
            return x_v, y_v, x2_v, y2_v, vehicle_id
    return -1, -1, -1, -1, -1

def read_license_plate(plate_img, reader):
    results = reader.readtext(plate_img)
    if results:
        text, confidence = results[0][1], results[0][2]
        return text, confidence
    return None, None

def write_csv(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Car_ID', 'License_Plate_Text', 'Confidence'])
        for frame_id, cars in results.items():
            for car_id, data in cars.items():
                writer.writerow([frame_id, car_id, data['license_plate']['text'], data['license_plate']['text_score']])

# Incarcam modelele
coco_model = YOLO('yolov8n.pt')  # Detectie vehicule
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # Detectie placute
reader = easyocr.Reader(['en'])  # OCR pentru citirea textului

# Initializam tracker-ul pentru vehicule
mot_tracker = Sort()

# Video input (foloseste camera)
cap = cv2.VideoCapture(0)  # 0 pentru camera implicită

# Clasele de vehicule dupa COCO (2=car, 3=motorcycle, 5=bus, 7=truck)
vehicles = [2, 3, 5, 7]

results = {}
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # Detectie vehicule
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Tracking vehicule
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = np.empty((0, 5))  # Returnează un array gol cu forma așteptată

    # Detectie placute
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Asociere placuta cu masina
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate[:5], track_ids)

        if car_id != -1:
            # Decupare placuta din imagine
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            # Convertim la grayscale pentru OCR
            license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Citire text placuta
            license_plate_text, license_plate_confidence = read_license_plate(license_plate_thresh, reader)

            if license_plate_text:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_confidence
                    }
                }

    # Afisare rezultatele pe ecran
    for car_id, data in results[frame_nmr].items():
        cv2.rectangle(frame, (int(data['license_plate']['bbox'][0]), int(data['license_plate']['bbox'][1])),
                      (int(data['license_plate']['bbox'][2]), int(data['license_plate']['bbox'][3])), (0, 255, 0), 2)
        cv2.putText(frame, data['license_plate']['text'],
                    (int(data['license_plate']['bbox'][0]), int(data['license_plate']['bbox'][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Scriem rezultatele in CSV
write_csv(results, './results.csv')

cap.release()
cv2.destroyAllWindows()