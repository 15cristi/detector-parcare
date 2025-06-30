import cv2
import easyocr
import numpy as np

# Incarcam imaginea placutei
image_path = "./plates/processed_11_4.0.jpg"  # Asigura-te ca numele fisierului este corect
image = cv2.imread(image_path)

if image is None:
    print("[ERROR] Nu s-a putut incarca imaginea pentru OCR!")
    exit()

# Convertim la grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicam CLAHE pentru imbunatatirea contrastului
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Aplicam un blur pentru reducerea zgomotului
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

# Aplicam thresholding binar
_, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

# Aplicam morfologie pentru curatare
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# ?? Scalare pentru OCR (marim imaginea de 2x)
morph = cv2.resize(morph, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

# Salvam imaginea preprocesata
cv2.imwrite("processed_plate_scaled.jpg", morph)

# Aplicam OCR
reader = easyocr.Reader(['en'], gpu=False)
detections = reader.readtext(morph)

# Afisam rezultatele OCR
print("\n[INFO] Rezultate OCR:")
for detection in detections:
    _, text, score = detection
    text = text.upper().replace(' ', '').replace('-', '').replace('_', '')
    print(f"Numar detectat: {text} (Scor: {score})")

# Afisam imaginea preprocesata marita
cv2.imshow("Processed Plate", morph)
cv2.waitKey(0)
cv2.destroyAllWindows()
