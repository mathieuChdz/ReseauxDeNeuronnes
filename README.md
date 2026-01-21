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

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="22"/> Prérequis (checkpoint)
 Prérequis (checkpoint)

> <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/exclamation-triangle.svg" width="20"/> Le script d’inférence nécessite le fichier **`vae_unet_best.pth`** (non inclus sur le repo).
> Place-le ici : `scripts/vae_unet_best.pth`

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
---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="22"/> Upscale de l’image restaurée


Un script d’upscaling basé sur **Real-ESRGAN** est fourni pour augmenter la résolution
de l’image restaurée (×4).

<img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/exclamation-triangle.svg" width="20"/> Ce script est **expérimental** : il améliore la résolution visuelle mais peut introduire des artefacts.

### Fonctionnement

- Entrée : `assets/before.jpg` (image restaurée en 128×128)
- Sortie : `assets/after.jpg` (image upscalée ~512×512)
---


> <img src="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/solid/information-circle.svg" width="20"/> L’upscaling est indépendant du modèle **VAE-UNet** et n’améliore pas
> les détails sémantiques, uniquement la résolution visuelle perçue.


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



















