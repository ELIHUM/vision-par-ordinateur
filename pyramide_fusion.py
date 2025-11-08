# ===========================================================
#   FUSION D'IMAGES PAR PYRAMIDE LAPLACIENNE
#   FUSION LAPLACIENNE POUR 2 IMAGES.
#   pour voir les niveau de fusion Laplacienne 4,5, 6
#   Il faut fermer l'image ceci passera au niveau 5 et ainsi de suite.
#
# ===========================================================

import cv2
import numpy as np
import os

# ===========================================================
# 1. Construction de la pyramide laplacienne
# ===========================================================

def build_laplacian_pyramid(image, levels):
    """Construit la pyramide laplacienne d'une image."""
    gp = [image.copy()]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)

    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        laplacian = cv2.subtract(gp[i], cv2.pyrUp(gp[i+1], dstsize=size))
        lp.append(laplacian)

    lp.append(gp[-1])  # Dernier niveau (base)
    return lp

# ===========================================================
# 2. Calcul des maximums pixel à pixel entre deux pyramides
# ===========================================================

def max_lap_pyr(lp1, lp2):
    """Fusionne deux pyramides laplaciennes par maximum pixel à pixel."""
    max_pyr = []
    for i in range(len(lp1) - 1):
        max_pyr.append(np.where(np.abs(lp1[i]) > np.abs(lp2[i]), lp1[i], lp2[i]))
    # Dernier niveau : moyenne
    max_pyr.append((lp1[-1].astype(np.float32) + lp2[-1].astype(np.float32)) / 2.0)
    return max_pyr

# ===========================================================
# 3. Reconstruction de l'image depuis une pyramide laplacienne
# ===========================================================

def reconstruct_from_lap_pyr(lp):
    """Reconstruit une image à partir d'une pyramide laplacienne."""
    image = lp[-1].astype(np.float32)
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        up = cv2.pyrUp(image, dstsize=size).astype(np.float32)
        image = cv2.add(up, lp[i].astype(np.float32))
    return np.clip(image, 0, 255).astype(np.uint8)
# ===========================================================
# 4. Fonction principale de fusion
# ===========================================================

def fus_lap_pyr(img1, img2, levels=5):
    """Fusionne deux images en utilisant la pyramide laplacienne."""
    # Redimensionner pour égaliser les tailles
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    lp1 = build_laplacian_pyramid(img1, levels)
    lp2 = build_laplacian_pyramid(img2, levels)

    fused_pyr = max_lap_pyr(lp1, lp2)
    fused_img = reconstruct_from_lap_pyr(fused_pyr)
    return np.clip(fused_img, 0, 255).astype(np.uint8)

# ===========================================================
# 6. Fusion Multi-Images (N images)
# ===========================================================

def fuseLap_N_Images(image_list, levels=5):
    """
    Fusionne N images en utilisant la méthode de la Pyramide Laplacienne.
    image_list : liste d'images (np.array)
    levels     : nombre de niveaux de la pyramide
    """

    # Vérification
    if len(image_list) < 2:
        raise ValueError("Il faut au moins 2 images pour la fusion.")

    # Assurer que toutes les images ont la même taille
    base_shape = (image_list[0].shape[1], image_list[0].shape[0])
    imgs = [cv2.resize(img, base_shape) for img in image_list]

    # 1️ Construction des pyramides Laplaciennes
    lap_pyrs = [build_laplacian_pyramid(img, levels) for img in imgs]

    # 2️ Fusion des niveaux (max magnitude pour chaque pixel)
    fused_pyr = []
    for level in range(levels):
        # Extraire le niveau 'level' de toutes les pyramides
        laps = [lap_pyrs[i][level].astype(np.float32) for i in range(len(imgs))]

        # Trouver les indices des valeurs max par pixel
        stacked = np.stack(laps, axis=-1)
        max_indices = np.argmax(np.abs(stacked), axis=-1)

        # Sélectionner les valeurs correspondantes
        fused_level = np.take_along_axis(stacked, np.expand_dims(max_indices, axis=-1), axis=-1)
        fused_pyr.append(fused_level.squeeze(-1))

    # 3️ Fusion du dernier niveau (moyenne des derniers niveaux gaussiens)
    last_levels = [lap_pyrs[i][-1].astype(np.float32) for i in range(len(imgs))]
    fused_last = np.mean(last_levels, axis=0)
    fused_pyr.append(fused_last)

    # 4️ Reconstruction
    fused_img = reconstruct_from_lap_pyr(fused_pyr)
    return fused_img


# ===========================================================
# 5. Test sur un dossier d'images
# ===========================================================

if __name__ == "__main__":
#C:\Users\elihu\OneDrive\Documents\projet_vision\test_fusion_2_images\test_fusion_2_images\set1\test11_claire.png
    niveaux = [4, 5, 6]

    print("\n==================== Résumé attendu ====================")
    print(" Objectif : Fusion d'images par Pyramide Laplacienne")
    print("--------------------------------------------------------")
    print(" JEU DE TEST : set1")
    print("--------------------------------------------------------")

#---------------------------------------------------------------------------------------------
    #test sur deux images précises du Set1
#---------------------------------------------------------------------------------------------

   
    dossier_test_1 = "test_fusion_2_images/test_fusion_2_images/set1"  # Dossier contenant les paires d'images de SET1
    
    # Exemple : test sur deux images précises du 
    img1_path = os.path.join(dossier_test_1, "test11_claire.png")
    img2_path = os.path.join(dossier_test_1, "test11_sombre.png")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
     
    if img1 is None or img2 is None:
        print(" Erreur : Impossible de charger les images. Vérifie les chemins.")
        exit()

    for level in niveaux:
       
        print(f"Fusion avec {level} niveaux de pyramide...")
        fused = fus_lap_pyr(img1, img2, levels=level)
        nom_sortie = f"fusion_laplacienne_1_{level}.png"
        cv2.imwrite(nom_sortie, fused)
        cv2.imshow(f"Fusion Laplacienne (Niveau {level})", fused)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    
    print("--------------------------------------------------------")
    print(" Analyse par niveau de pyramide :")
    print(" • Niveau 4 → transitions visibles mais amélioration du contraste global.")
    print(" • Niveau 5 → bon équilibre entre luminosité et netteté (recommandé).")
    print(" • Niveau 6 → rendu plus doux, mais léger flou dans les contours fins.")
    print("--------------------------------------------------------")
    print(" Sorties générées :")
    print(" - fusion_laplacienne_1_4.png")
    print(" - fusion_laplacienne_1_5.png")
    print(" - fusion_laplacienne_1_6.png")
    print("Ces fichiers se trouvent dans ton dossier projet_vision.")
    print("--------------------------------------------------------")

#---------------------------------------------------------------------------------------------
#                       test sur set2
#---------------------------------------------------------------------------------------------

    print(" JEU DE TEST : set2")
    print("--------------------------------------------------------")

    dossier_test_2 = "test_fusion_2_images/test_fusion_2_images/set2"
    print("test sur deux images précises du Set2")
    # Exemple : test sur deux images précises du 
    img1_path = os.path.join(dossier_test_2, "test11_flou_leger.png")
    img2_path = os.path.join(dossier_test_2, "test11.png")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(" Erreur : Impossible de charger les images. Vérifie les chemins.")
        exit()

    for level in niveaux:
       
        print(f"Fusion avec {level} niveaux de pyramide...")
        fused = fus_lap_pyr(img1, img2, levels=level)
        nom_sortie = f"fusion_laplacienne_2_{level}.png"
        cv2.imwrite(nom_sortie, fused)
        cv2.imshow(f"Fusion Laplacienne (Niveau {level})", fused)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    
    print("--------------------------------------------------------")
    print("Analyse par niveau de pyramide :")
    print(" • Niveau 4 → Lissage perceptible, contours légèrement adoucis.")
    print(" • Niveau 5 → Bon compromis entre netteté et suppression du bruit.")
    print(" • Niveau 6 → Image plus douce, mais perte légère de détails fins.")
    print("--------------------------------------------------------")
    print(" Sorties générées :")
    print(" - fusion_laplacienne_2_4.png")
    print(" - fusion_laplacienne_2_5.png")
    print(" - fusion_laplacienne_2_6.png")
    print("Les fichiers se trouvent dans ton dossier projet_vision (set2).")

    print(" Exécution terminée. Consulte les images générées dans ton dossier projet_vision.")
    print("========================================================\n\n")

    #=======================================================================================================================
    # Dossier contenant les images à fusionner (N images)
    #=======================================================================================================================
    
    

    print("FUSION N IMAGE : set1N")
    print("========================================================\n")

    # c'est ici ou se trouve le chemin des image 
    dossier_test_1 = "test_fusion_N_images/test_fusion_N_images/set1N"
    #niveaux = [4, 5, 6]

    # Charger toutes les images du dossier
    image_list = []
    for filename in sorted(os.listdir(dossier_test_1)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(dossier_test_1, filename)
            img = cv2.imread(path)
            if img is not None:
                image_list.append(img)

    print(f"{len(image_list)} images chargées pour la fusion N-images.")

    if len(image_list) < 2:
        print(" Pas assez d'images pour la fusion.")
        exit()

    # Tester la fusion pour plusieurs niveaux
    for level in niveaux:
        print(f"Fusion multi-images avec {level} niveaux de pyramide...")
        fused = fuseLap_N_Images(image_list, levels=level)
        nom_sortie = f"fusion_N_images_1_{level}.png"
        cv2.imwrite(nom_sortie, fused)
        cv2.imshow(f"Fusion N Images (Niveau {level})", fused)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(" Analyse par niveau de pyramide :")
    print(" • Niveau 4 → Bonne visibilité des détails mais contrastes marqués.")
    print(" • Niveau 5 → Équilibre idéal entre contraste, luminosité et texture.")
    print(" • Niveau 6 → Image plus douce, très naturelle (effet HDR perceptible).")
    print("--------------------------------------------------------")
    print(" Sorties générées :")
    print(" - fusion_N_images_1_4.png")
    print(" - fusion_N_images_1_5.png")
    print(" - fusion_N_images_1_6.png")
    print("Ces fichiers se trouvent dans ton dossier projet_vision (set1N).")
    print("==================================================================\n")

#---------------------------------------------------------------------------------------
#                           set_2_N
#---------------------------------------------------------------------------------------


    print("FUSION N IMAGE : set2N")
    print("========================================================\n")

    # c'est ici ou se trouve le chemin des image 
    dossier_test_2 = "test_fusion_N_images/test_fusion_N_images/set2N"
    #niveaux = [4, 5, 6]

    # Charger toutes les images du dossier
    image_list = []
    for filename in sorted(os.listdir(dossier_test_2)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(dossier_test_2, filename)
            img = cv2.imread(path)
            if img is not None:
                image_list.append(img)

    print(f"{len(image_list)} images chargées pour la fusion N-images.")

    if len(image_list) < 2:
        print(" Pas assez d'images pour la fusion.")
        exit()

    # Tester la fusion pour plusieurs niveaux
    for level in niveaux:
        print(f"Fusion multi-images avec {level} niveaux de pyramide...")
        fused = fuseLap_N_Images(image_list, levels=level)
        nom_sortie = f"fusion_N_images_2_{level}.png"
        cv2.imwrite(nom_sortie, fused)
        cv2.imshow(f"Fusion N Images (Niveau {level})", fused)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(" Analyse par niveau de pyramide :")
    print(" • Niveau 4 → Bonne visibilité des détails mais contrastes marqués.")
    print(" • Niveau 5 → Équilibre idéal entre contraste, luminosité et texture.")
    print(" • Niveau 6 → Image plus douce, très naturelle (effet HDR perceptible).")
    print("--------------------------------------------------------")
    print(" Sorties générées :")
    print(" - fusion_N_images_1_4.png")
    print(" - fusion_N_images_1_5.png")
    print(" - fusion_N_images_1_6.png")
    print("Ces fichiers se trouvent dans ton dossier projet_vision (set1N).")
    print("========================================================\n")
    print(" Exécution terminée. Consulte les images générées dans ton dossier projet_vision.")