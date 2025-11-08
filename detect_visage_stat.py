import cv2
import sys
import operator

# Étape 1 – Chargement et préparation de l'image
image_path = "C:/Users/locmu/OneDrive/Documents/Courses/4eme/Info4304/Devoir/Tp1/ff.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Erreur : impossible de charger l'image. Vérifie le chemin :", image_path)
    sys.exit()

# Conversion en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Affichage de l'image d'origine pour vérifier son chargement
cv2.imshow("Image d'origine", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Étape 2 – Chargement des classifieurs Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

if face_cascade.empty() or profile_cascade.empty():
    print("Erreur : impossible de charger les fichiers cascade XML.")
    sys.exit()

# Détection des visages frontaux
faces_front = face_cascade.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=8, minSize=(80, 80)
)

# Détection des profils de visage
faces_profile = profile_cascade.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=8, minSize=(80, 80)
)

# Transformation géométrique (symétrie horizontale) et détection des profils
gray_flipped = cv2.flip(gray, 1)
faces_profile_flipped = profile_cascade.detectMultiScale(
    gray_flipped, scaleFactor=1.3, minNeighbors=8, minSize=(80, 80)
)

# Correction des coordonnées x après le flip horizontal
img_width = gray.shape[1]
faces_profile_flipped_corrected = [
    (img_width - x - w, y, w, h) for (x, y, w, h) in faces_profile_flipped
]

# Étape 3 – Traitement et affichage des résultats
# Concaténation de toutes les détections dans une seule liste
list_faces = list(faces_front) + list(faces_profile) + faces_profile_flipped_corrected

# Dessin d'un rectangle rouge autour de chaque visage détecté
for (x, y, w, h) in list_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)

# Affichage du nombre total de visages détectés
print(f"Nombre total de visages détectés : {len(list_faces)}")

# Étape 4 – Amélioration de la visualisation
# Redimensionnement de l'image pour l'affichage (réduction à 25%)
height, width = image.shape[:2]
image_display = cv2.resize(image, (width // 4, height // 4))

# Affichage de l'image annotée
cv2.imshow("Visages détectés", image_display)

# Sauvegarde de l'image avec les détections
cv2.imwrite("result_detect_faces.png", image)

cv2.waitKey(0)
cv2.destroyAllWindows()