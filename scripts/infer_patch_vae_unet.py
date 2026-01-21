import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

# ------------ CONFIGURATION ------------
IMAGE_INPUT_PATH = "scripts\\before2.jpg"  # Image à corriger
IMAGE_OUTPUT_PATH = "assets\\after2.jpg"
CHECKPOINT_PATH = "scripts\\vae_unet_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ ARCHITECTURE DU MODÈLE (DOIT ÊTRE IDENTIQUE) ------------
class VAE_UNet(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE_UNet, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = self.conv_block(3, 64); self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = self.conv_block(64, 128); self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = self.conv_block(128, 256); self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = self.conv_block(256, 512); self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 1024 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (1024, 8, 8))
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, 3, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        
        # Bottleneck & Latent
        b = self.bottleneck(p4)
        flat = self.flatten(b)
        mu, logvar = self.fc_mu(flat), self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.unflatten(self.fc_decode(z))
        d4 = torch.cat([self.up4(x), e4], dim=1); d4 = self.dec4(d4)
        d3 = torch.cat([self.up3(d4), e3], dim=1); d3 = self.dec3(d3)
        d2 = torch.cat([self.up2(d3), e2], dim=1); d2 = self.dec2(d2)
        d1 = torch.cat([self.up1(d2), e1], dim=1); d1 = self.dec1(d1)
        
        return torch.tanh(self.out(d1))

# ------------ CHARGEMENT ET INFERENCE ------------

def run_inference():
    # 1. Initialisation du modèle
    model = VAE_UNet(latent_dim=64).to(DEVICE)
    
    # 2. Chargement des poids
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Erreur : Le fichier {CHECKPOINT_PATH} est introuvable.")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # On charge le state_dict (vérifie si c'est la structure complète ou juste les poids)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Modèle chargé avec succès.")

    # 3. Préparation de l'image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Recalage [-1, 1]
    ])

    if not os.path.exists(IMAGE_INPUT_PATH):
        print(f"Erreur : L'image {IMAGE_INPUT_PATH} est introuvable.")
        return

    img = Image.open(IMAGE_INPUT_PATH).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 4. Inférence
    print("🚀 Restauration en cours...")
    with torch.no_grad():
        restored_tensor = model(input_tensor)

    # 5. Dé-normalisation et Sauvegarde
    # On repasse de [-1, 1] à [0, 1]
    restored_tensor = (restored_tensor.squeeze(0).cpu() * 0.5) + 0.5
    restored_tensor = torch.clamp(restored_tensor, 0, 1)

    save_image(restored_tensor, IMAGE_OUTPUT_PATH)
    print(f"✨ Image restaurée (128x128) sauvegardée ici : {IMAGE_OUTPUT_PATH}")

if __name__ == "__main__":
    run_inference()