import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# --- 1. RECOPIE DE L'ARCHITECTURE DU GÉNÉRATEUR ---
# (Obligatoire pour charger les poids du .pth)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True))
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 128, 3, 1, 1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True))
        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 64, 3, 1, 1), nn.InstanceNorm2d(64, affine=True), nn.ReLU(True))
        self.u3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2)
        u1 = self.u1(d3); u1 = torch.cat([u1, d2], dim=1)
        u2 = self.u2(u1); u2 = torch.cat([u2, d1], dim=1)
        return self.u3(u2)

# --- 2. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On récupère le dossier où se trouve le script (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# On remonte d'un cran pour arriver à la racine (Reseauxdeneurones/)
root_dir = os.path.dirname(current_dir)

# --- CORRECTION DES CHEMINS ---
# models/checkpoints_gan/gan_best.pth
model_path = os.path.join(root_dir, "models", "checkpoints_gan", "gan_best.pth")

# assets/nom_de_ton_image.jpg
input_image_path = os.path.join(root_dir, "assets", "degraded_image_000001.jpg") 

# assets/resultat_restauration.png
output_path = os.path.join(root_dir, "assets", "resultat_restauration.png")

# --- 3. CHARGEMENT DU MODÈLE ---
gen = Generator().to(device)
if os.path.exists(model_path):
    # Utilisation de weights_only=False pour éviter l'erreur UnpicklingError
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()
    print(f"Modèle chargé : {model_path}")
else:
    print("Erreur : Fichier .pth introuvable !")
    exit()

# --- 4. PRÉPARATION DE L'IMAGE ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Mise en [-1, 1]
])

img = Image.open(input_image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# --- 5. INFÉRENCE (TRANSFORMATION) ---
with torch.no_grad():
    output = gen(input_tensor)

# --- 6. DÉ-NORMALISATION ET SAUVEGARDE ---
# [-1, 1] -> [0, 1]
output = (output.squeeze(0).cpu() * 0.5) + 0.5
output = output.clamp(0, 1)

# Conversion en image PIL et sauvegarde
res_img = transforms.ToPILImage()(output)
res_img.save(output_path)
print(f"Image restaurée sauvegardée sous : {output_path}")