import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

# ------------ CONFIGURATION ------------
IMAGE_INPUT_PATH = "data\\degraded_images\\degraded_image_00999.jpg"
IMAGE_OUTPUT_PATH = "data\\after.jpg"
CHECKPOINT_PATH = "models\\GAN\\checkpoints\\checkpoint_epoch_60.pth"  # <-- ton .pth avec gen_state_dict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------ GENERATOR (IDENTIQUE À L'ENTRAINEMENT) ------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True)
        )

        self.u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True)
        )
        self.u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        )
        self.u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)

        u1 = self.u1(d3)
        u1 = torch.cat([u1, d2], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d1], dim=1)

        return self.u3(u2)


# ------------ INFERENCE ------------
def run_inference():
    # 1) Init modèle
    gen = Generator().to(DEVICE)

    # 2) Charger checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Erreur : Le fichier {CHECKPOINT_PATH} est introuvable.")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Ton training sauvegarde "gen_state_dict"
    if isinstance(checkpoint, dict) and "gen_state_dict" in checkpoint:
        state_dict = checkpoint["gen_state_dict"]
    else:
        # fallback: si tu as sauvegardé directement gen.state_dict()
        state_dict = checkpoint

    gen.load_state_dict(state_dict, strict=True)
    gen.eval()
    print("✅ Générateur chargé (GAN).")

    # 3) Préparation image (doit matcher le training)
    # Normalisation vers [-1, 1] car sortie gen = Tanh
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if not os.path.exists(IMAGE_INPUT_PATH):
        print(f"Erreur : L'image {IMAGE_INPUT_PATH} est introuvable.")
        return

    img = Image.open(IMAGE_INPUT_PATH).convert("RGB")
    degraded = transform(img).unsqueeze(0).to(DEVICE)

    # 4) Inférence
    print("🚀 Restauration GAN en cours...")
    with torch.no_grad():
        restored = gen(degraded)

    # 5) Dé-normalisation : [-1,1] -> [0,1]
    restored = (restored.squeeze(0).cpu() + 1.0) / 2.0
    restored = torch.clamp(restored, 0, 1)

    save_image(restored, IMAGE_OUTPUT_PATH)
    print(f"✨ Image restaurée sauvegardée ici : {IMAGE_OUTPUT_PATH}")


if __name__ == "__main__":
    run_inference()
