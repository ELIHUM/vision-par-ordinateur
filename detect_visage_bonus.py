import cv2
import numpy as np

# === Paramètres du modèle ===
model_path = "opencv_face_detector_uint8.pb"
config_path = "opencv_face_detector.pbtxt"

# Charger le modèle DNN
print(" Chargement du modèle DNN...")
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
print(" Modèle chargé avec succès.")

# Initialisation de la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Erreur : impossible d'accéder à la webcam.")
    exit()

print(" Webcam activée. Appuyez sur 'q' pour quitter.")

# === Paramètres de détection ===
conf_threshold = 0.7  # seuil de confiance
input_size = (300, 300)
mean_values = (104, 117, 123)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Impossible de lire la frame.")
        break

    # Redimensionnement pour accélérer le traitement
    frame = cv2.resize(frame, (640, 480))
    (h, w) = frame.shape[:2]

    # Préparation du blob pour le modèle DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, input_size, mean_values, swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    faces = []

    # Parcours des détections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            faces.append((x1, y1, x2 - x1, y2 - y1, confidence))

    # Dessin des détections
    for i, (x, y, width, height, conf) in enumerate(faces, start=1):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = f"Visage {i} ({conf*100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Affichage du nombre total de visages détectés
    cv2.putText(frame, f"{len(faces)} visage(s) détecté(s)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Affichage vidéo
    cv2.imshow("Détection de visages (Appuyez sur Q pour quitter)", frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
print(" Programme terminé.")
