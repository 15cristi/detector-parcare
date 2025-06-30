import easyocr
import re
import csv
import cv2
import numpy as np

# Initializam OCR
reader = easyocr.Reader(['en'], gpu=False)

def read_license_plate(image_path):
    """Citim textul de pe placuta si returnam textul + scorul OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("[ERROR] Nu s-a putut incarca imaginea pentru OCR!")
        return None, None

    detections = reader.readtext(image)

    if not detections:
        return None, None

    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '').replace('-', '').replace('_', '')

        # Eliminam caracterele invalide
        text = re.sub(r'[^A-Z0-9]', '', text)

        if is_valid_plate(text):
            return text, score

    return None, None

def is_valid_plate(text):
    text = text.replace(' ', '').replace('-', '')
    pattern = r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$"
    return re.match(pattern, text) is not None


def get_car(license_plate, vehicle_track_ids):
    """
    Asociaza placuta de inmatriculare cu o masina.
    Compara coordonatele placutei cu coordonatele masinilor detectate.
    """
    x1, y1, x2, y2, _, _ = license_plate

    for track in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track

        # Verificam daca placuta este in interiorul masinii detectate
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1

# def write_csv(results, output_path):
#     """Scrie rezultatele detectate in fisierul CSV."""
#     with open(output_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["license_number"])

#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 if 'license_plate' in results[frame_nmr][car_id]:
#                     license_plate_text = results[frame_nmr][car_id]['license_plate']['text']
#                     writer.writerow([license_plate_text])

#     print(f"[INFO] Datele au fost salvate in {output_path}")

