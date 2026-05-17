# 🏛️ Archival Restoration Suite

> Pipeline complet de restauration numérique d'images d'archives — Version Console & GUI

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

---

## Table des matières

- [Présentation](#présentation)
- [Architecture du projet](#architecture-du-projet)
- [Fonctionnalités détaillées](#fonctionnalités-détaillées)
- [Prérequis et installation](#prérequis-et-installation)
- [Utilisation — Mode Console (`xRestore2a.py`)](#utilisation--mode-console-xrestore2apy)
- [Utilisation — Mode GUI (`xRestoreTK.py`)](#utilisation--mode-gui-xrestorettkpy)
- [Référence complète des paramètres](#référence-complète-des-paramètres)
- [Modules techniques](#modules-techniques)
- [Exemples d'usage avancé](#exemples-dusage-avancé)
- [Rapport de restauration JSON](#rapport-de-restauration-json)
- [Métriques de qualité](#métriques-de-qualité)
- [Dépendances optionnelles et fallbacks](#dépendances-optionnelles-et-fallbacks)
- [Limitations connues](#limitations-connues)
- [Contribuer](#contribuer)

---

## Présentation

**Archival Restoration Suite** est un outil Python de restauration numérique d'images d'archives photographiques, conçu pour traiter les dégradations typiques du patrimoine photographique (albums anciens, daguerréotypes, tirages papier, négatifs film, etc.).

Le projet se compose de **deux fichiers complémentaires** :

| Fichier | Version | Usage |
|---|---|---|
| `xRestore2a.py` | Console v3.0 | Pipeline en ligne de commande, scriptable, intégrable dans des workflows automatisés |
| `xRestoreTK.py` | GUI v4.0 | Interface graphique Tkinter avec prévisualisation en temps réel et contrôles interactifs |

Les deux partagent le même moteur de traitement et les mêmes algorithmes. La version console est privilégiée pour le traitement par lots ou les environnements sans affichage, tandis que la version GUI est idéale pour un travail interactif et l'ajustement fin des paramètres.

---

## Architecture du projet

```
archival-restoration-suite/
│
├── xRestore2a.py          # Version console (CLI) — pipeline complet
├── xRestoreTK.py          # Version GUI (Tkinter) — interface interactive
│
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances Python (à créer, voir ci-dessous)
│
└── examples/              # (optionnel) Images d'exemple avant/après
    ├── sample_input.jpg
    └── sample_output.png
```

### Modules partagés entre les deux versions

Les deux fichiers embarquent les mêmes classes de traitement :

```
DegradationAnalyzer       ← Détection automatique des types de dégradation
SpectralRestorer          ← Filtrage homomorphique, congruence de phase, diffusion anisotrope
InpaintingEngine          ← Inpainting Telea, Navier-Stokes, par patchs exemplaires
ColorScience              ← Balance des blancs, récupération des couleurs fanées, déjaunissement
DeconvolutionRestorer     ← Estimation PSF aveugle, Richardson-Lucy, déconvolution Wiener
SuperResolution           ← Lanczos, rétroprojection itérative, dirigée par contours
NoiseReduction            ← BM3D, NLM, ondelettes, bilatéral
ContrastEnhancer          ← CLAHE, Retinex MSRCR, Reinhard TMO
GeometryCorrector         ← Redressement auto, correction distorsion
DamageRestorer            ← Foxing, moisissure, craquelures, déchirures, dégâts d'eau, daguerréotype
VintageEffects            ← Vignettage, grain film, halation, effet délavé
QualityMetrics            ← Netteté, contraste, colorimétrie, SNR
```

---

## Fonctionnalités détaillées

### Détection automatique des dégradations

L'analyseur inspecte l'image et identifie jusqu'à 18 types de dégradation :

| Type | Détection | Description |
|---|---|---|
| `FADING` | Luminosité moyenne élevée, faible écart-type | Image décolorée, voile général |
| `YELLOWING` | Déséquilibre canal rouge+vert vs bleu | Jaunissement caractéristique du papier acide |
| `NOISE` | Estimation sigma (skimage) | Bruit de grain ou numérique |
| `BLUR` | Gradient Sobel faible | Image floue ou mise au point manquée |
| `SCRATCHES` | Transformée de Hough (lignes droites) | Rayures linéaires nombreuses |
| `FOXING` | Détection HSV (tons brun-rouille) | Taches de rousseur caractéristiques du vieillissement du papier |
| `VIGNETTING` | Comparaison centre vs coins | Assombrissement aux bords |
| `BANDING` | Analyse FFT des lignes | Bandes horizontales répétitives |
| `COLOR_SHIFT` | Déséquilibre inter-canaux RGB | Dominante colorée parasite |
| `SILVERING` | Détection HSV (zones claires, faible saturation) | Mirage argentique sur tirages anciens |
| `WATER_DAMAGE` | Variance locale élevée | Auréoles et traces d'humidité |
| `CRACKS` | Composantes connexes allongées sombres | Craquelures de l'émulsion ou du support |
| `TEARS` | Zones claires linéaires | Déchirures du support |
| `MOLD` | Détection HSV (tons verts) | Moisissures biologiques |
| `OXIDATION` | (daguerréotype) Zones sombres locales | Oxydation de surface argentée |

### Restauration des dommages

Chaque type de dommage dispose d'un traitement dédié basé sur la segmentation HSV et l'inpainting :

- **Foxing** : isolation par masque HSV brun-rouille + inpainting Telea
- **Moisissure** : masque HSV vert + inpainting Navier-Stokes
- **Craquelures** : analyse des composantes connexes sombres allongées + inpainting Telea
- **Déchirures** : détection de zones claires linéaires (morphologie) + inpainting par patchs
- **Dégâts d'eau** : lissage bilatéral sélectif sur les zones à variance locale élevée
- **Mirage argentique** : atténuation des zones à haute valeur HSV et faible saturation
- **Daguerréotype** : correction CLAHE locale de l'oxydation + lissage tarnishing

### Inpainting

Trois méthodes disponibles :

| Méthode | Algorithme | Usage |
|---|---|---|
| `telea` | Fast Marching Method (Telea 2004) | Rapide, idéal pour les petites zones |
| `ns` | Équations de Navier-Stokes (Bertalmio et al.) | Meilleure cohérence texturale |
| `exemplar` | Inpainting par patchs SSD | Zones larges, textures répétitives |

### Réduction de bruit

| Méthode | Bibliothèque | Caractéristiques |
|---|---|---|
| `bm3d` | skimage | Meilleure qualité, plus lent |
| `nlm` | skimage / OpenCV | Non-local Means, bon compromis |
| `wavelet` | skimage | Rapide, préserve bien les bords |
| `bilateral` | OpenCV | Très rapide, filtrage par empilement |

### Amélioration du contraste

| Méthode | Algorithme | Usage |
|---|---|---|
| `clahe` | CLAHE en espace LAB | Usage général, préserve la teinte |
| `retinex` | MSRCR multi-échelle | Images à fort contraste local |
| `reinhard` | Tone Mapping Opérateur | Images HDR ou à grande dynamique |

### Science des couleurs

- **Gray World** : hypothèse que la moyenne de toutes les couleurs tend vers le gris
- **White Patch** : correction basée sur le point le plus lumineux (99.9e percentile)
- **Shades of Gray** : généralisation de Gray World avec exposant de Minkowski (p=6)
- **Récupération de décoloration** : remontée par correction gamma canal par canal
- **Déjaunissement** : soustraction vectorielle de la composante jaune dans l'espace RGB

### Super-résolution

| Méthode | Facteurs | Description |
|---|---|---|
| `lanczos` | 2×, 3×, 4× | Interpolation Lanczos (référence qualité) |
| `ibp` | 2×, 3×, 4× | Rétroprojection itérative (5 itérations, propagation d'erreur) |
| `edge` | 2×, 3×, 4× | Dirigée par contours avec filtrage bilatéral sur zones d'arrêtes |

### Traitement spectral

- **Filtre homomorphique** : séparation illumination/réflectance dans le domaine fréquentiel (filtre Butterworth d'ordre 2)
- **Netteté par congruence de phase** : accentuation basée sur la mesure de cohérence de phase locale (invariant au contraste)
- **Diffusion anisotrope de Perona-Malik** : débruitage préservant les contours par équations différentielles partielles (15 itérations, kappa=30)

### Effets vintage

- **Vignettage** : masque radial paramétrable (intensité + forme)
- **Grain film** : bruit gaussien lumineux ou coloré
- **Halation** : diffusion rougeâtre des hautes lumières (halo photographique)
- **Effet délavé** : fusion avec un gris moyen

---

## Prérequis et installation

### Version Python

Python **3.8 ou supérieur** est requis.

### Installation des dépendances

```bash
# Dépendances principales (obligatoires)
pip install opencv-python numpy pillow scipy

# Dépendances optionnelles (recommandées pour une qualité maximale)
pip install scikit-image scikit-learn

# Pour la version GUI (Tkinter est généralement inclus avec Python)
# Sur Ubuntu/Debian si absent :
sudo apt-get install python3-tk

# Installation complète en une commande
pip install opencv-python numpy pillow scipy scikit-image scikit-learn
```

### Fichier `requirements.txt` suggéré

```
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=9.0.0
scipy>=1.7.0
scikit-image>=0.19.0
scikit-learn>=1.0.0
```

### Vérification de l'installation

```bash
python -c "import cv2, numpy, PIL, scipy; print('Dépendances principales OK')"
python -c "import skimage, sklearn; print('Dépendances optionnelles OK')"
```

---

## Utilisation — Mode Console (`xRestore2a.py`)

### Syntaxe générale

```bash
python xRestore2a.py <image_source> [options]
```

### Démarrage rapide

```bash
# Restauration archivale complète (mode par défaut)
python xRestore2a.py photo_ancienne.jpg

# Restauration rapide sans débruitage lourd
python xRestore2a.py photo.png --mode quick

# Spécifier un fichier de sortie
python xRestore2a.py photo.jpg -o restauree.png
```

### Exemples par cas d'usage

```bash
# Album photo jauni — déjaunissement + récupération couleurs + netteté
python xRestore2a.py album.jpg \
    --remove-yellowing \
    --color-fade-recovery \
    --sharpen-method unsharp \
    --sharpen-amount 0.8

# Daguerréotype oxydé
python xRestore2a.py daguerreotype.jpg \
    --medium daguerreotype \
    --oxidation-removal \
    --denoise-method nlm \
    --contrast-method clahe

# Photo avec foxing et moisissures
python xRestore2a.py archive.png \
    --restore-foxing \
    --foxing-aggressiveness 0.9 \
    --restore-mold \
    --compare

# Super-résolution 2× + débruitage BM3D
python xRestore2a.py photo_basse_res.jpg \
    --superres 2 \
    --superres-method ibp \
    --denoise-method bm3d

# Analyse spectrale complète avec rapport JSON
python xRestore2a.py photo.png \
    --mode archival \
    --spectral-analysis \
    --spectral-bands 16 \
    --report-json rapport.json

# Image de négatif nitrate film
python xRestore2a.py nitrate.jpg \
    --medium nitrate_film \
    --restore-all-damage \
    --contrast-method retinex \
    --denoise-method wavelet

# Inpainting manuel avec masque externe
python xRestore2a.py photo.jpg \
    --inpaint-mask masque.png \
    --inpaint-method exemplar \
    --inpaint-radius 7

# Mode custom avec contrôle fin
python xRestore2a.py photo.jpg \
    --mode custom \
    --gamma 1.2 \
    --brightness 1.1 \
    --saturation 1.3 \
    --warmth 20 \
    --vignette 0.3 \
    --film-grain 0.02

# Génération d'un comparatif avant/après
python xRestore2a.py photo.jpg \
    --mode archival \
    --compare \
    --compare-output comparaison.png \
    --verbose
```

### Flux d'exécution du pipeline console

```
Chargement image
     ↓
Analyse des dégradations (DegradationAnalyzer)
     ↓
[1] Correction géométrique (rotation, recadrage, redressement, distorsion)
     ↓
[2] Super-résolution (Lanczos / IBP / Edge-directed)
     ↓
[3] Restauration dommages (foxing, moisissure, craquelures, déchirures, eau, mirage, daguerréotype)
     ↓
[4] Inpainting manuel (si masque fourni)
     ↓
[5] Débruitage (BM3D / NLM / Wavelet / Bilatéral)
     ↓
[6] Contraste (CLAHE / Retinex / Reinhard)
     ↓
[7] Science des couleurs (constance + décoloration + jaunissement + chaleur + sépia)
     ↓
[8] Netteté (Unsharp / Congruence de phase / Déconvolution Richardson-Lucy)
     ↓
[9] Traitements spectraux (Homomorphique / Diffusion anisotrope)
     ↓
[10] Effets vintage (vignettage, grain, halation, fade)
     ↓
[11] Ajustements finaux (luminosité, gamma, ombres, hautes lumières, saturation)
     ↓
Métriques qualité avant/après
     ↓
Sauvegarde image + rapport JSON
```

---

## Utilisation — Mode GUI (`xRestoreTK.py`)

### Lancement

```bash
python xRestoreTK.py
```

### Interface

L'interface se divise en deux zones :

**Panneau gauche** — Contrôles organisés en onglets :
| Onglet | Contenu |
|---|---|
| 📐 Géométrie | Redressement auto, distorsion lentille, rotation, échelle |
| 🔍 Super-résolution | Facteur (×1 à ×4), méthode |
| 🏛️ Dommages | Cases à cocher pour chaque type de dégradation + aggressivité foxing |
| 🔇 Débruitage | Activation, méthode, sigma |
| 🌓 Contraste | Méthode, paramètres CLAHE, gamma, luminosité, ombres, hautes lumières |
| 🎨 Couleur | Constance couleur, récupération, déjaunissement, saturation, chaleur, sépia |
| 🔎 Netteté | Activation, méthode, intensité, rayon, itérations |
| 🔬 Spectral | Filtre homomorphique, diffusion anisotrope |
| 🎞️ Vintage | Vignettage, grain, halation, effet délavé |

**Panneau droit** — Zone de prévisualisation :
- Canvas scrollable avec zoom (10% à 500%)
- Ajustement automatique à la fenêtre
- Mise à jour automatique différée (debounce 300ms) à chaque modification d'un paramètre
- Traitement dans un thread séparé (interface non bloquante)

### Barre d'outils

| Bouton | Raccourci clavier | Action |
|---|---|---|
| 📂 Ouvrir | `Ctrl+O` | Charger une image (PNG, JPG, BMP, TIFF) |
| 💾 Sauvegarder | `Ctrl+S` | Sauvegarder à l'emplacement d'origine |
| 💾 Sauvegarder sous... | — | Choisir un nouvel emplacement |
| ↩ Annuler | `Ctrl+Z` | Retour à l'état précédent (20 niveaux) |
| ↪ Refaire | `Ctrl+Y` | Rétablir une action annulée |
| 🔄 Réinitialiser | — | Retour à l'image originale (avec confirmation) |
| ⚡ Appliquer | `F5` | Sauvegarder l'état courant dans l'historique |
| 📊 Métriques | — | Afficher les métriques avant/après |
| 🔬 Analyse | — | Lancer l'analyse automatique des dégradations |
| ➖ / ➕ Zoom | `Ctrl+-` / `Ctrl++` | Zoom avant/arrière |
| Ajuster | `Ctrl+0` | Ajuster l'image à la fenêtre |

### Boutons d'actions rapides

- **🎯 Mode Archival Complet** : active automatiquement les options recommandées (redressement, BM3D, CLAHE, Shades of Gray, unsharp mask, foxing, moisissure, craquelures, déjaunissement, récupération couleurs)
- **🔄 Appliquer tous les changements** : sauvegarde l'état courant dans la pile d'annulation et relance le pipeline

---

## Référence complète des paramètres

### Entrée / Sortie

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `input` | positional | — | Chemin de l'image source |
| `-o` / `--output` | str | `<input>_restauree.png` | Chemin de sortie |
| `--compare` | flag | False | Génère une image comparatif avant/après côte à côte |
| `--compare-output` | str | `<output>_comparaison.png` | Chemin du comparatif |
| `--quality` | int 1-100 | 95 | Qualité JPEG en sortie |
| `--report-json` | str | None | Chemin du rapport JSON de restauration |
| `-v` / `--verbose` | flag | False | Affichage détaillé du traitement |

### Mode de restauration

| Paramètre | Valeurs | Défaut | Description |
|---|---|---|---|
| `--mode` | `archival`, `quick`, `custom` | `archival` | `archival` = pipeline complet, `quick` = rapide sans débruitage lourd, `custom` = contrôle manuel total |
| `--medium` | `paper_bw`, `paper_color`, `daguerreotype`, `tintype`, `ambrotype`, `glass_plate`, `nitrate_film`, `acetate_film`, `polaroid`, `chromogenic` | `paper_color` | Type de support photographique (adapte les traitements) |

### Géométrie

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--correct-geometry` | flag | False | Active la correction géométrique automatique |
| `--auto-straighten` | flag | False | Redressement automatique par transformée de Hough |
| `--lens-distortion` | float | None | Coefficient de distorsion radiale k1 (négatif = distorsion en barillet) |
| `--rotate` | float | 0.0 | Rotation en degrés (sens horaire positif) |
| `--scale` | float | 1.0 | Facteur de redimensionnement global |
| `--crop` | 4×int | None | Recadrage : `X Y W H` en pixels |

### Super-résolution

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--superres` | int {1,2,3,4} | 1 | Facteur d'agrandissement |
| `--superres-method` | `lanczos`, `ibp`, `edge` | `ibp` | Méthode d'interpolation |

### Restauration de dommages

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--restore-foxing` | flag | False | Suppression des taches de foxing |
| `--foxing-aggressiveness` | float 0-1 | 0.7 | Aggressivité de la détection foxing |
| `--restore-mold` | flag | False | Suppression des moisissures |
| `--restore-cracks` | flag | False | Réparation des craquelures |
| `--restore-tears` | flag | False | Réparation des déchirures |
| `--restore-water` | flag | False | Correction des dégâts d'eau |
| `--restore-silvering` | flag | False | Suppression du mirage argentique |
| `--oxidation-removal` | flag | False | Suppression de l'oxydation (daguerréotype) |
| `--restore-all-damage` | flag | False | Active toutes les restaurations de dommages |

### Débruitage

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--denoise-method` | `bm3d`, `nlm`, `wavelet`, `bilateral`, `none` | `bm3d` | Algorithme de débruitage |
| `--denoise-sigma` | float | auto | Sigma du bruit (estimé automatiquement si absent) |
| `--no-denoise` | flag | False | Désactive entièrement le débruitage |

### Contraste & Luminosité

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--contrast-method` | `clahe`, `retinex`, `reinhard`, `none` | `clahe` | Algorithme d'amélioration du contraste |
| `--clahe-clip` | float | 2.0 | Limite d'amplification CLAHE (plus élevé = plus contrasté) |
| `--clahe-grid` | int | 8 | Taille de la grille de tuiles CLAHE |
| `--retinex-sigmas` | float+ | [15, 80, 250] | Sigmas gaussiens pour Retinex multi-échelle |
| `--gamma` | float | None | Correction gamma (< 1 = éclaircir, > 1 = assombrir) |
| `--brightness` | float | 1.0 | Facteur de luminosité globale |
| `--shadows-lift` | float 0-1 | 0.0 | Relève les zones sombres |
| `--highlights-compress` | float 0-1 | 0.0 | Comprime les hautes lumières |

### Couleur

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--color-constancy` | `gray_world`, `white_patch`, `shades_gray`, `none` | `shades_gray` | Algorithme de balance des blancs automatique |
| `--color-fade-recovery` | flag | False | Récupération des couleurs décolorées |
| `--remove-yellowing` | flag | False | Suppression du jaunissement |
| `--yellowing-strength` | float | 1.0 | Intensité du déjaunissement |
| `--saturation` | float | 1.0 | Facteur de saturation (1.0 = inchangé) |
| `--warmth` | float -100/+100 | 0.0 | Température de couleur (positif = chaud/orange, négatif = froid/bleu) |
| `--sepia` | float 0-1 | 0.0 | Intensité de l'effet sépia |

### Netteté

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--sharpen-method` | `unsharp`, `phase_congruency`, `deconvolution`, `none` | `unsharp` | Algorithme d'accentuation |
| `--sharpen-amount` | float | 0.6 | Intensité de l'accentuation |
| `--sharpen-radius` | float | 1.2 | Rayon du flou pour unsharp mask |
| `--deconv-iterations` | int | 8 | Nombre d'itérations Richardson-Lucy |
| `--no-sharpen` | flag | False | Désactive entièrement la netteté |

### Analyse spectrale

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--spectral-analysis` | flag | False | Active l'analyse spectrale fréquentielle |
| `--spectral-bands` | int | 8 | Nombre de bandes spectrales pour l'analyse |
| `--homomorphic-filter` | flag | False | Active le filtre homomorphique |
| `--anisotropic-diffusion` | flag | False | Active la diffusion anisotrope de Perona-Malik |

### Inpainting manuel

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--inpaint-mask` | str | None | Chemin vers un masque d'inpainting (blanc = zones à reconstruire) |
| `--inpaint-method` | `telea`, `ns`, `exemplar` | `telea` | Méthode d'inpainting |
| `--inpaint-radius` | int | 5 | Rayon de propagation de l'inpainting |

### Effets vintage

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `--vignette` | float 0-1 | 0.0 | Intensité du vignettage |
| `--vignette-shape` | float | 2.0 | Forme du vignettage (2.0 = elliptique, valeurs plus élevées = plus carré) |
| `--film-grain` | float 0-0.1 | 0.0 | Intensité du grain film |
| `--grain-color` | flag | False | Grain en couleur (plutôt que lumineux uniquement) |
| `--halation` | float 0-1 | 0.0 | Intensité de l'halation |
| `--fade` | float 0-0.5 | 0.0 | Intensité de l'effet délavé |

---

## Modules techniques

### `DegradationAnalyzer`

```python
analyzer = DegradationAnalyzer(img_np)  # img_np : tableau NumPy RGB uint8
degradations = analyzer.analyze_all()    # → List[DegradationType]
spectral = analyzer.get_spectral_signature(n_bands=8)  # → Dict
```

Chaque méthode de détection est indépendante et peut être appelée séparément (`_detect_foxing()`, `_detect_blur()`, etc.).

### `InpaintingEngine`

```python
# Inpainting Telea (Fast Marching)
result = InpaintingEngine.telea_inpaint(img, mask, radius=5)

# Inpainting Navier-Stokes
result = InpaintingEngine.ns_inpaint(img, mask, radius=5)

# Inpainting par patchs exemplaires (plus lent, meilleure qualité sur grandes zones)
result = InpaintingEngine.exemplar_based_inpaint(img, mask, patch_size=11)
```

Le masque peut être uint8 (0/255) ou float (0.0/1.0), les deux formats sont acceptés.

### `DeconvolutionRestorer`

```python
# Estimation aveugle de la PSF (fonction d'étalement du point)
psf = DeconvolutionRestorer.estimate_psf_blind(img, kernel_size=15)

# Déconvolution Richardson-Lucy (itérative)
sharpened = DeconvolutionRestorer.richardson_lucy(img, psf, iterations=10)

# Déconvolution de Wiener (non-itérative, plus rapide)
sharpened = DeconvolutionRestorer.wiener_deconvolution(img, psf, k=0.01)
```

### `QualityMetrics`

```python
metrics = QualityMetrics.evaluate(img)
# Retourne :
# {
#   "sharpness":    float,  # Variance du Laplacien (plus élevé = plus net)
#   "contrast":     float,  # Écart-type sur le canal gris
#   "colorfulness": float,  # Métrique de Hasler & Süsstrunk
#   "snr_db":       float   # Rapport signal/bruit en dB
# }
```

---

## Exemples d'usage avancé

### Traitement par lot d'un dossier

```bash
#!/bin/bash
# batch_restore.sh
for f in archives/*.jpg; do
    echo "Traitement : $f"
    python xRestore2a.py "$f" \
        --mode archival \
        --remove-yellowing \
        --color-fade-recovery \
        --report-json "rapports/$(basename $f .jpg).json" \
        --compare
done
```

### Intégration en tant que module Python

```python
import sys
sys.argv = [
    'xRestore2a.py',
    'photo.jpg',
    '--mode', 'archival',
    '--remove-yellowing',
    '--restore-foxing',
    '--output', 'resultat.png'
]

from xRestore2a import build_parser, ArchivalRestorationPipeline

parser = build_parser()
args = parser.parse_args()
pipeline = ArchivalRestorationPipeline(args)
report = pipeline.process(args.input, args.output)

print(f"Temps total : {report.processing_time:.2f}s")
print(f"Dégradations : {report.detected_degradations}")
```

### Inpainting avec masque personnalisé

Le masque doit être une image en niveaux de gris où **blanc (255) = zones à reconstruire**, noir (0) = zones à conserver.

```bash
# Créer un masque avec GIMP ou Photoshop, puis :
python xRestore2a.py photo.jpg \
    --inpaint-mask masque_retouche.png \
    --inpaint-method exemplar \
    --inpaint-radius 11
```

---

## Rapport de restauration JSON

En mode `archival`, un fichier JSON est automatiquement généré. En mode `custom` ou `quick`, utilisez `--report-json chemin.json`.

### Structure du rapport

```json
{
  "input_path": "photo.jpg",
  "output_path": "photo_restauree.png",
  "original_dimensions": [2400, 3200],
  "final_dimensions": [2400, 3200],
  "detected_degradations": ["YELLOWING", "NOISE", "FOXING"],
  "applied_steps": [
    {"name": "Correction géométrique", "duration": 0.12},
    {"name": "Suppression foxing", "duration": 1.45},
    {"name": "Débruitage (BM3D)", "duration": 8.32},
    {"name": "Contraste (clahe)", "duration": 0.08},
    {"name": "Constance couleur (shades_gray)", "duration": 0.04},
    {"name": "Netteté (unsharp)", "duration": 0.06}
  ],
  "spectral_analysis": {
    "band_0": {"mean": 0.021, "std": 0.015},
    "band_1": {"mean": 0.048, "std": 0.031}
  },
  "quality_metrics": {
    "before": {
      "sharpness": 145.3,
      "contrast": 42.1,
      "colorfulness": 18.7,
      "snr_db": 28.4
    },
    "after": {
      "sharpness": 312.8,
      "contrast": 56.9,
      "colorfulness": 31.2,
      "snr_db": 34.1
    }
  },
  "processing_time": 12.47
}
```

---

## Métriques de qualité

| Métrique | Calcul | Interprétation |
|---|---|---|
| **Sharpness** | Variance du Laplacien sur canal gris | Plus élevée = plus nette. Valeurs typiques : < 100 (flou), 100-500 (correct), > 500 (très net) |
| **Contrast** | Écart-type de l'histogramme gris | Plus élevée = meilleur contraste. < 30 (faible), 30-60 (correct), > 60 (bon) |
| **Colorfulness** | Métrique de Hasler & Süsstrunk (2003) | Plus élevée = couleurs plus vives. < 10 (quasi-gris), 10-40 (modéré), > 40 (vivace) |
| **SNR (dB)** | Rapport signal/bruit (puissance) | Plus élevé = moins de bruit. > 30 dB (bon), 20-30 dB (acceptable), < 20 dB (bruité) |

---

## Dépendances optionnelles et fallbacks

Le projet est conçu pour fonctionner même sans toutes les bibliothèques installées :

| Module | Si absent | Impact |
|---|---|---|
| `scikit-image` | Fallback OpenCV / scipy | Débruitage NLM moins efficace, pas de BM3D, pas d'estimation sigma automatique |
| `scikit-learn` | Non utilisé activement | Pas d'impact sur les fonctionnalités principales |
| `tkinter` | — | Uniquement requis pour `xRestoreTK.py` |

Les fonctions critiques testent systématiquement `SKIMAGE_AVAILABLE` avant d'appeler scikit-image et basculent sur des équivalents OpenCV/scipy le cas échéant.

---

## Limitations connues

- **Inpainting par patchs exemplaires** : algorithme en Python pur, lent sur grandes zones (préférer `telea` ou `ns` pour les performances)
- **BM3D** : disponible via `skimage.restoration.denoise_bm3d` uniquement depuis scikit-image >= 0.19 ; peut ne pas être disponible sur toutes les installations
- **Super-résolution** : les méthodes disponibles sont purement algorithmiques (pas de réseau de neurones) ; pour une super-résolution de haute qualité, envisager Real-ESRGAN ou BSRGAN
- **Analyse de dégradations** : les seuils de détection sont calibrés pour des photographies d'archives typiques ; des images très atypiques peuvent produire des faux positifs ou des faux négatifs
- **Daguerréotype** : la restauration est une approximation algorithmique ; les daguerréotypes physiques nécessitent un traitement chimique spécialisé pour une restauration complète
- **Performance GUI** : le traitement est lancé dans un thread séparé mais le rendu Tkinter reste sur le thread principal ; sur des images très grandes (> 50 Mpx), l'interface peut sembler lente

---

## Contribuer

Les contributions sont les bienvenues. Quelques pistes d'amélioration :

- Intégration de modèles de super-résolution basés sur des réseaux de neurones (ESRGAN, SwinIR)
- Support du traitement par lots dans l'interface GUI
- Export de presets de restauration (JSON)
- Histogramme interactif dans la GUI
- Support des formats RAW (via `rawpy`)
- Tests unitaires pour chaque module de traitement

Pour proposer une modification : fork → branche dédiée → Pull Request avec description des changements et, si possible, exemples visuels avant/après.

---

*Archival Restoration Suite — Laboratoire de Restauration Numérique*
