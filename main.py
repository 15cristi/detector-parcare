from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
import re
from sort import *
from util import get_car, read_license_plate, write_csv

# Funcție pentru validarea unui număr de înmatriculare

def is_valid_plate(plate_text):
    pattern = r"^[A-Z]{1,2}[0-9]{2,3}[A-Z]{3}$"  # Exemplu pentru numere românești
    return re.match(pattern, plate_text) is not None

# Inițializare cameră cu libcamera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

results = {}
mot_tracker = Sort()

# Încarcă modelele YOLO
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

vehicles = [2, 3, 5, 7]  # Clase de vehicule (ex: mașini, camioane)
frame_nmr = -1
from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')  # Model pentru detectarea vehiculelor
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # Model pentru detectarea plăcuțelor de înmatriculare

# Load video
cap = cv2.VideoCapture(0)  # Video din cameră (poți schimba cu calea către un fișier video)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # Fereastră pentru afișare
vehicles = [2, 3, 5, 7]  # Clase de vehicule (de exemplu, mașini, camioane etc.)

# Read frames
frame_nmr = -1
ret = True
while ret:
    
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera", frame)  # Afișează imaginea live
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Apasă 'q' pentru a ieși
            break
        # print(f"Procesare cadru {frame_nmr}...")  # Debug pentru a vedea pe care cadru suntem
        results[frame_nmr] = {}
        
        # Detectează vehicule
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        print(f"Vehicule detectate: {detections_}")  # Debug pentru detecțiile vehiculelor

        # Verificăm dacă există vehicule detectate înainte de a actualiza tracker-ul
        if detections_:
            # Urmărește vehiculele
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detectează plăcuțele de înmatriculare
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # print(f"Plăcuță de înmatriculare detectată: {x1}, {y1}, {x2}, {y2}, Scor: {score}")  # Debug pentru plăcuțele de înmatriculare

                # Asociază plăcuța de înmatriculare cu un vehicul
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:

                    # Taie plăcuța de înmatriculare din cadrul imaginii
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # Procesează plăcuța de înmatriculare (în alb-negru și binar)
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Citește textul plăcuței de înmatriculare
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    print(f"Text OCR detectat: {license_plate_text}, Scor OCR: {license_plate_text_score}")  # Debug pentru textul OCR

                    # Verifică scorul OCR înainte de a salva rezultatele
                    if license_plate_text is not None and license_plate_text_score > 0.2:  # Filtrăm doar scorurile mai mari de 0.6
                        print(f"Salvăm rezultatele pentru {license_plate_text}")  # Debug pentru confirmare că datele vor fi salvate
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
                    else:
                        print(f"Plăcuța {license_plate_text} nu este validă (scor < 0.6).")
            
            # Scrie rezultatele în fișier CSV
            if len(results) > 0:
                #print("Salvăm în CSV...")  # Debug pentru salvarea în CSV
                write_csv(results, './test.csv')
        # else:
        #    #print(f"Nu au fost detectate vehicule pe cadru {frame_nmr}.") 

cap.release()
cv2.destroyAllWindows()
while True:
    frame_nmr += 1
    
    # Capturează un cadru din cameră
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    results[frame_nmr] = {}

    # Detectează vehicule
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    
    if detections_:
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        # Detectează plăcuțele de înmatriculare
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Aplică thresholding adaptiv
                license_plate_crop_thresh = cv2.adaptiveThreshold(
                    license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                )
                
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text and is_valid_plate(license_plate_text) and license_plate_text_score > 0.01:
                    print(f"Număr detectat: {license_plate_text}, Scor OCR: {license_plate_text_score}")
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                
                # Afișează imaginea procesată
                cv2.imshow("Plăcuță procesată", license_plate_crop_thresh)
                
        if results:
            write_csv(results, './test.csv')

# Cleanup
cv2.destroyAllWindows()
picam2.stop()