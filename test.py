import cv2
import easyocr
import re

# Lista județelor din România
judete_romania = {"AB", "AR", "AG", "BC", "BH", "BN", "BR", "BT", "BV", "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN"}

# Inițializare EasyOCR
# reader = easyocr.Reader(['en', 'ro'])  # Adăugat limba română
reader = easyocr.Reader(['en'], gpu = True)

# Deschidem fluxul video de la Iriun (camerele web sunt de obicei la index 0 sau 1)
cap = cv2.VideoCapture(0)  # Poți schimba cu `0` dacă camera nu este la index 1

if not cap.isOpened():
    print("Eroare la deschiderea camerei!")
    exit()

# Funcție de corectare a erorilor OCR (de exemplu, schimbarea L în 1, O în 0)
def corectare_text(text):
    text = text.replace("L", "1").replace("O", "0")
    return text

while True:
    placuta_detectata=False
    # Citim un frame din video
    ret, frame = cap.read()
    
    if not ret:
        print("Eroare la capturarea imaginii!")
        break

    # Conversie la gri pentru îmbunătățirea performanței OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Crește contrastul pentru a ajuta OCR
    alpha = 3.0  # Factor de contrast
    beta = 60    # Factor de luminozitate
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Aplică un blur pentru a reduce zgomotul
    gray = cv2.GaussianBlur(adjusted, (5, 5), 0)
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Detectare contururi pentru a găsi plăcuța
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filtrăm contururile mici și neimportante
        if area > 495:  # Prag de dimensiune pentru zona plăcuței
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            # Verificăm raportul lățime/înălțime (proporție similară cu a unei plăcuțe de înmatriculare)
            if 2 < aspect_ratio < 6:
                placuta = frame[y:y+h, x:x+w]  # Crop pe zona detectată
                placuta_detectata = True
                break  # Oprește după prima plăcuță validă

    if placuta_detectata:
        # Aplicăm OCR pe zona cropată
        results = reader.readtext(placuta)
        if not results:
            print("Nu s-a detectat niciun text în această zonă.")
        
        possible_numbers = []
        for (_, text, prob) in results:
            print(f"Text detectat: {text} - Precizie: {prob:.2f}")
            # Curățăm textul pentru a elimina caracterele inutile
            text = text.upper().replace(" ", "").replace(";", "").replace("RO", "")
            text = corectare_text(text)  # Corectăm textul detectat
            # Filtrăm textul pentru a păstra doar ce începe cu județul
            if len(text) > 5 and text[:2] in judete_romania:
                # Înlocuim orice caracter înainte de județ
                numar_de_inmatriculare = text
                possible_numbers.append((numar_de_inmatriculare, prob))
        
        if possible_numbers:
            # Selectăm numărul cu probabilitatea cea mai mare
            numar_detectat = max(possible_numbers, key=lambda x: x[1])
            print(f"Număr de înmatriculare detectat: {numar_detectat[0]}")
            cv2.putText(frame, f"Detectat: {numar_detectat[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dreptunghi pe imaginea originală
            # Face imaginea gri
            placuta_gray = cv2.cvtColor(placuta, cv2.COLOR_BGR2GRAY)
            
            # Aplicăm din nou OCR pe imaginea gri pentru a citi numărul cu spații
            gray_results = reader.readtext(placuta_gray)
            if gray_results:
                # Extragem textul din rezultatele OCR
                final_text = " ".join([text for (_, text, _) in gray_results]).strip()
                # Curățăm și corectăm textul final citit
                final_text = corectare_text(final_text)
                print(f"Text final citit: {final_text}")
                # Salvăm imaginea gri cu numărul de înmatriculare citit
                
            
            break  # Oprește procesul după ce numărul este salvat

    # Afișează imaginea procesată
    cv2.imshow("Detectare Număr Inmatriculare", frame)

    # Ieșire din buclă la apăsarea tastelor 'q' sau 'esc'
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:  # 27 = ESC
        break

# Închidem captura video și feroniza resursele
cap.release()
cv2.destroyAllWindows()