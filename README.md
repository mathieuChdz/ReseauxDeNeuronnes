# ReseauxDeNeuronnes
Projet Reseaux de Neuronnes M2 S1

<!-- ========================================================= -->
<!--                         VAE-UNet                           -->
<!--        Image Restoration Inference (PyTorch, 128x128)       -->
<!-- ========================================================= -->

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=900&center=true&vCenter=true&width=900&lines=VAE-UNet+%F0%9F%8E%AF+Image+Restoration;Inference+Script+%E2%9A%A1+PyTorch;Restore+degraded+images+to+128x128+%F0%9F%96%BC%EF%B8%8F" alt="Typing SVG" />
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="TorchVision" src="https://img.shields.io/badge/TorchVision-yes-5C2D91"></a>
  <a href="#"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-auto--detect-76B900?logo=nvidia&logoColor=white"></a>
  <a href="#"><img alt="OS" src="https://img.shields.io/badge/OS-Windows%2FLinux%2FmacOS-2ea44f"></a>
</p>

<p align="center">
  <b>Script d'inférence</b> pour restaurer une image dégradée avec un modèle <code>VAE_UNet</code> (auto-detect GPU/CPU), sauvegarde une image restaurée en sortie.
</p>

---

# Contexte

Dans ce projet, nous adoptons trois approches distinctes basées sur des modèles génératifs : les Variational Autoencoders (VAE), privilégiant une reconstruction stable et cohérente, et les Generative Adversarial Networks (GAN), orientés vers la génération d’images ainsi que les réseaux de neurones convolutifs (CNN). Cette comparaison permettra d’analyser les avantages et les limites de chaque modèle dans le cadre de la restauration et de la colorisation d’images.

L'objectif ici est de savoir, pour notre cas, quel est le meilleur modèle.

Enfin, une dernière partie (bonus) pourra être dédiée à l’upscaling, dont l’objectif est d’augmenter la résolution des images restaurées afin d’améliorer leur qualité perceptuelle. 

# Partie 0 : Organisation du projet

Une bonne organisation est nécéssaire pour une meilleure compréhension.


```text
.
├── data/
├── models/
│   └── VAE/
│        └── checkpoints/
│        └── output_vae/
│        └── vae_model.ipynb/
│   └── GAN/
│        └── checkpoints/
│        └── output_gan/
│        └── gan_model.ipynb/
│   └── CNN/
│        └── checkpoints/
│        └── output_cnn/
│        └── cnn_model.ipynb/
├── scripts/
│   └── preprocess_dataset.ipynb/
│   └── testVae.py/
│   └── testGan.py/
│   └── testCnn.py/
├── requirements.txt
└── README.md

```

# Partie 1 : Restauration d'images

<div align="center">
  <br />

  <p align="center">
  <code style="color: #ff5555; font-size: 1.2em;"><b>Accès direct au test de Restauration (GAN + VAE + HD Upscale)</b></code>
</p>

  <div align="center">
  <br />
  <a href="https://colab.research.google.com/drive/1aqXILdJ1NoTIZO8X2fN-VoZoep5-uSW1?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="250px">
  </a>
  <p><i>Cliquez ici pour tester nos modèles GAN, VAE et CNN en temps réel</i></p>
</div>

  **Pourquoi tester notre Colab ?**
  *Aucune installation requise* • *GPU T4 gratuit* • *Pipeline complet (128px ➡️ 512px HD)*
  
  <br />

  ### **Modèles pré-entraînés (Poids .pth)**
  Si vous préférez l'exécution locale, téléchargez nos meilleurs checkpoints :

  [![GAN Weights](https://img.shields.io/badge/Model-GAN_Best-FFee83?style=for-the-badge&logo=google-drive&logoColor=black)](https://drive.google.com/file/d/1qr1Gj9-yr_U8-8676P4Oft9xFY2B3Y42/view?usp=sharing)
  [![VAE Weights](https://img.shields.io/badge/Model-VAE_Best-FAC898?style=for-the-badge&logo=google-drive&logoColor=black)](https://drive.google.com/file/d/17ON3jYFBf_VgBfE8d3ZpI23QxL5kkZ8t/view?usp=sharing)[![CNN Weights](https://img.shields.io/badge/Model-CNN_Best-FAC898?style=for-the-badge&logo=google-drive&logoColor=black)](https://drive.google.com/file/d/1Wx7S1T8_-anwkrkQvld8F0LXliBpH97w/view?usp=sharing)

  <br />
</div>

---

## <span style="color:#FAC898">Dataset</span>

Le dataset utilisé est un dataset de ~60 000 images aléatoires. Un prétraitement a été effectué sur celles-ci pour les détériorer.

Lien du dataset (pris sur kaggle) : https://www.kaggle.com/datasets/starktony45/image-dataset/data

Les nouvelles images ont les caractériques suivantes :
* Redimensionnement --> 128x128
* Passage de RGB à grayscale (1 canal)
* Ajout de bruits aléatoires : gaussien, sel et poivre, multiplicatif, corrélé, structuré
* Application d’un flou aléatoire
* Compression JPEG

### <span style="color:#39B52F">Exemple</span> : prétraitement image de base (haut) vs image dégradée (bas)

<table align="center">
  <tr>
    <th>Image 1</th>
    <th>Image 2</th>
    <th>Image 3</th>
    <th>Image 4</th>
    <th>Image 5</th>
    <th>Image 6</th>
  </tr>
  <tr>
    <td><img src="assets/image_dataset_1.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_2.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_3.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_4.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_5.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_6.jpg" width="96"/></td>
  </tr>
  <tr>
    <td><img src="assets/image_dataset_deg1.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_deg2.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_deg3.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_deg4.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_deg5.jpg" width="96"/></td>
    <td><img src="assets/image_dataset_deg6.jpg" width="96"/></td>
  </tr>
</table>

## <span style="color:#FAC898">Approche VAE</span>

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="22"/> Prérequis (checkpoint)
 Prérequis (checkpoint)

> <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/exclamation-triangle.svg" width="20"/> Le script d’inférence nécessite le fichier **`vae_unet_best.pth`** (non inclus sur le repo).
> Place-le ici : `scripts/vae_unet_best.pth`

## Explication du modèle

Voici une proposition de modification pour ta section **Approche VAE**, rédigée avec le même style visuel et le même niveau de détail que ta section GAN :

---

### <span style="color:#FAC898">Approche VAE</span>

## Explication du modèle

Le modèle repose sur une architecture hybride combinant la modélisation probabiliste des VAE et la précision des U-Net.

### Architecture <span style="color:#FFee83">VAE-UNet</span> :

* **Encodeur** :
  Blocs convolutifs (Conv + BatchNorm + LeakyReLU) extrayant les caractéristiques latentes.


* **Bottleneck (Espace Latent)** :
  Apprentissage des vecteurs $\mu$ (moyenne) et $\sigma$ (écart-type). Utilisation du **Reparameterization Trick** pour permettre la backpropagation.


* **Décodeur** :
  Reconstruction par convolutions transposées avec Skip-connections pour préserver les textures fines de l'image source.




### Training :

L'entraînement du VAE repose sur l'optimisation de deux objectifs complémentaires via une fonction de perte composite :

#### 1. Perte de Reconstruction (MSE ou L1)

* Mesure la fidélité de l'image restaurée par rapport à l'image cible (ground truth).
* Force le décodeur à reconstruire des images visuellement cohérentes.

#### 2. Divergence de Kullback-Leibler (KLD)

* Régularise l'espace latent en forçant la distribution apprise à se rapprocher d'une distribution normale standard .
* Évite le sur-apprentissage et assure une structure continue dans l'espace latent.

#### 3. Optimisation

* Utilisation de l'optimiseur **Adam**.
* Ajustement dynamique du poids de la KLD (-VAE) pour équilibrer la netteté de l'image et la régularité du modèle.

---


## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="22"/> Utilisation simple (Avant → Après)
 Utilisation simple (Avant → Après)

1. Ajouter une image dégradée nommée **`before.jpg`** dans le dossier `assets/`.
2. Lancer le script d’inférence.
3. L’image restaurée est générée automatiquement sous le nom **`after.jpg`** dans `assets/`.

<img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/arrow-right.svg" width="20"/>
 Le modèle travaille en résolution **128×128**.


## <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/sparkles.svg" width="20"/> Démo rapide (TL;DR)

```bash
# 1) Installer les dépendances
pip install -r requirements.txt

# 2) Lancer l'inférence
python scripts/infer_patch_vae_unet.py
```


## <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/photo.svg" width="20"/> Résultats — Avant / Après

<table align="center">
  <tr>
    <th>Avant ❌</th>
    <th>Après ✅</th>
    <th>Avant ❌</th>
    <th>Après ✅</th>
    <th>Avant ❌</th>
    <th>Après ✅</th>
  </tr>
  <tr>
    <td><img src="assets/before1.jpg" width="96"/></td>
    <td><img src="assets/after1.jpg"  width="96"/></td>
    <td><img src="assets/before2.jpg" width="96"/></td>
    <td><img src="assets/after2.jpg"  width="96"/></td>
    <td><img src="assets/before3.jpg" width="96"/></td>
    <td><img src="assets/after3.jpg"  width="96"/></td>
  </tr>
  <tr>
    <td><img src="assets/before4.jpg" width="96"/></td>
    <td><img src="assets/after4.jpg"  width="96"/></td>
    <td><img src="assets/before5.jpg" width="96"/></td>
    <td><img src="assets/after5.jpg"  width="96"/></td>
  </tr>
</table>



## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linux/linux-original.svg" width="22"/> Limites connues


- <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/user.svg" width="20"/> Le modèle fonctionne **nettement mieux lorsqu’une seule personne** est présente dans l’image.
- <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/users.svg" width="20"/> Les performances diminuent lorsque **plusieurs personnes** apparaissent simultanément.
- <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/photo.svg" width="20"/> La résolution est limitée à **128×128 pixels**, ce qui peut entraîner :
  - une perte de détails fins,
  - des artefacts sur les visages ou les contours complexes.
- <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/cpu-chip.svg" width="20"/> Ces limites sont liées :
  - à la capacité du modèle,
  - à la résolution d’entraînement,
  - et à la distribution des données d’apprentissage.


---

## <span style="color:#FAC898">Approche GAN</span>

## Explication du modèle

### Générateur <span style="color:#FFee83">G</span> :

* Réseau encodeur–décodeur de type U-Net

* **Encodeur** :
  * 3 couches convolutionnels qui réduit la résolution

* **Décodeur** :
  * 3 couches d'upsampling qui reconstruisent l’image

  * Des skip-connections sont utilisées pour diminuer la perte de détails (dans les couches intermédiaires)

### Discriminateur <span style="color:#FFee83">D</span> :

* CNN conditionnel --> évalue si une image est réelle ou générée

**Principe** :
* L'image dégradée et la réelle (ou la générée) sont données en input.
* Plusieurs couches convolutionnelles sont faites.
* Renvoie un 'panel/carte' de score indiquant le réalisme local de l’image.
  
### Training :

L’entraînement alterne entre le Discriminateur et le Générateur.

À chaque époque, l’entraînement se déroule comme suit :

#### 1. Discriminateur (1 mise à jour)
* Le discriminateur est entraîné une fois pour distinguer les images réelles des images générées.

#### 2. Générateur (2 mises à jour)
Le générateur est entraîné deux fois afin de :
* **Tromper le discriminateur** (perte adversariale).
* **Rester proche de l’image cible** (perte L1).
* **Améliorer la qualité perceptuelle** (perte VGG).

#### 3. Ajustement du learning rate
Les taux d’apprentissage du générateur et du discriminateur sont mis à jour à la fin de chaque époque.

## <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/sparkles.svg" width="20"/> Démo / Test du modèle

```bash
# 1) Installer les dépendances
  pip install -r requirements.txt

# 2) Dans le script testGAN.py, renseigner :
  IMAGE_INPUT_PATH = "chemin de l'image à améliorer"
  IMAGE_OUTPUT_PATH = "data\\after.jpg"
  CHECKPOINT_PATH = "chemin du .pth à utiliser" 

# 2) Lancer le test
  python scripts/testGan.py
```

---

## <span style="color:#FAC898">Approche CNN</span>
---
On utilise une approche basée sur les reseaux convolutifs, deux CNN distincts unepour traiter les images dégradées et un autre pour les colorier.
l'objectif principal est d’obtenir une restauration fidèle et stable, en particulier au niveau des couleurs, via une pipeline déterministe
# Vue de la pipeline
méthode repose sur deux réseaux CNN successifs, avec chacun un rôle distinct :
1. CNN1 : restauration structurelle de l’image dégradée

2. CNN2 : reconstruction des couleurs à partir de la luminance

# CNN1 : Reseau de Restauration 

Le premier est utilisé pour corriger les dégradations globales (bruit, peu de contraste, etc). Ca nous permet d'obtenir une image structurellement cohérente, mais avec des couleurs tres limitées. Afin de pallier ce problème un second réseau est introduit pour se concentrer spécifiquement sur la reconstruction de la couleur.

Il prend en entrée une image RGB normalisée dans l’espace [-1,1] et prédit une image restaurée dans le même espace.


Ce réseau est entraîné avec une fonction de perte sévère, melange de :

1. L1 : reconstruction de pixels stable
2. MSE ! penalise des erreurs fortes
3. Perceptual loss (VGG16) : amelioration visuelle (textures, details, ...)

afin de garantir une reconstruction  stable.

Ce qu'on gagne ne terme de texture / neteté on pert en couleurs, avec un aspet désaturé et fades (sepia / noir & blanc)


#  CNN : Reseau de coloration

Le second réseau est un U-Net dédié à la reconstruction des couleurs.

1. Entrée : luminance extraite de l’image restaurée par CNN1
2. Sortie : image RGB complète
3. Architecture : U-Net (encodeur/décodeur avec skip connections) pour capturer le context et conservation des details

Predir a partir de la luminance permet d'eviter que le reseau ne previligie les solutions grises et mieux preserver la saturation


Les skip connections assurent la conservation des détails spatiaux tout au long du réseau et facilitent la propagation de l’information de couleur.


# Fonction de Loss 
l'’entraînement du réseau de colorisation repose principalement sur une perte L1 entre l’image prédite et l’image de référence.

Ce choix est motivé par :

. une meilleure stabilité que la perte L2

. une réduction des artefacts

. une meilleure préservation des couleurs

Des pertes perceptuelles basées sur des réseaux pré-entraînés (VGG) ont été envisagées, mais leur coût computationnel élevé sur CPU a limité leur utilisation dans la version finale.

#Résultats et discussino
bla bla bla sur la restauration effective etc, etc

Limites:
. incapacité de gerer les details
. coloration trop extreme ou extremement fade (parler des autres versions entamées avec une loss plus généreuse qui genere des images entierment en rouge etc

...

---

# Partie 2 : Upscale

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="22"/> (BONUS) Upscale de l’image restaurée -- Algo récupérer sur un gitgub


### Résolution x2 x4

Ce premier upscale est tiré d'un dépôt git et nous permet de comprendre cela fonctionne et avoir un premier résultat.

Lien du dépôt : https://github.com/sberbank-ai/Real-ESRGAN.git

Un script d’upscaling basé sur **Real-ESRGAN** est fourni pour augmenter la résolution de l’image restaurée (×4).

<img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/exclamation-triangle.svg" width="20"/> Ce script est **expérimental** : il améliore la résolution visuelle mais peut introduire des artefacts.

### Fonctionnement

- Entrée : `assets/before.jpg` (image restaurée en 128×128)
- Sortie : `assets/after.jpg` (image upscalée ~512×512)
---


> <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/information-circle.svg" width="20"/> L’upscaling est indépendant du modèle **VAE-UNet** et n’améliore pas
> les détails sémantiques, uniquement la résolution visuelle perçue.

### Résolution taille initiale
















