import easyocr
import re

# Initializam OCR
reader = easyocr.Reader(['en'], gpu=False)

def read_license_plate(license_plate_crop):
    """Citeste textul de pe placuta si returneaza textul + scorul OCR."""
    detections = reader.readtext(license_plate_crop)

    if not detections:
        print("[WARNING] OCR nu a detectat niciun text!")
        return None, None

    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '').replace('-', '').replace('_', '')

        print(f"[DEBUG] OCR a detectat: {text} (Scor: {score})")  # Debugging

        # Filtram caracterele speciale si verificam lungimea
        text = re.sub(r'[^A-Z0-9]', '', text)  

        if is_valid_plate(text):
            return text, score

    print("[WARNING] OCR a detectat un text invalid!")
    return None, None

def is_valid_plate(text):
    """Verifica daca textul respecta formatul unui numar romanesc."""
    pattern = r"^[A-Z]{1,2}[0-9]{2,3}[A-Z]{3}$"  # Ex: B156BOS, MH14WOW
    return re.match(pattern, text) is not None

def get_car(license_plate, vehicle_track_ids):
    """
    Asociaza placuta de inmatriculare cu o masina.
    Compara coordonatele placutei cu coordonatele masinilor detectate.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for track in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track

        # Verificam daca placuta este in interiorul masinii detectate
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    """Scrie rezultatele detectate in fisierul CSV."""
    with open(output_path, 'w') as f:
        f.write("license_number\n")

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                license_plate_text = results[frame_nmr][car_id]['license_plate']['text']
                f.write(f"{license_plate_text}\n")

    print(f"[INFO] Datele au fost salvate in {output_path}")
