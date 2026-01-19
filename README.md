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

## ✨ Démo rapide (TL;DR)

```bash
# 1) Installer les dépendances
pip install -r requirements.txt

# 2) Lancer l'inférence
python scripts/inference.py
