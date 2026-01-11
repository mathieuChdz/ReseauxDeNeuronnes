import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ------------ CONFIG -------------
IMAGE_PATH = "scripts/input.jpg"
OUTPUT_PATH = "scripts/output_128.png"
CHECKPOINT_PATH = "scripts/vae_unet_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------ MODEL --------------
class VAE_UNet(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc1 = self.conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(512, 1024)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 1024 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (1024, 8, 8))

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
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
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        b = self.bottleneck(p4)
        flat = self.flatten(b)
        mu, logvar = self.fc_mu(flat), self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        x = self.unflatten(self.fc_decode(z))
        x = self.dec4(torch.cat([self.up4(x), e4], 1))
        x = self.dec3(torch.cat([self.up3(x), e3], 1))
        x = self.dec2(torch.cat([self.up2(x), e2], 1))
        x = self.dec1(torch.cat([self.up1(x), e1], 1))

        return torch.tanh(self.out(x)), mu, logvar


# ------------ LOAD MODEL ----------
model = VAE_UNet(latent_dim=64).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ------------ IMAGE PREP ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)  # [-1,1]
])

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ------------ INFERENCE -----------
with torch.no_grad():
    restored, _, _ = model(img_tensor)

# ------------ DE-NORMALIZE --------
restored = restored.squeeze(0).cpu()
restored = (restored * 0.5) + 0.5  # [-1,1] → [0,1]
restored = torch.clamp(restored, 0, 1)

# ------------ SAVE ----------------
save_image(restored, OUTPUT_PATH)
print(f"✅ Image restaurée en 128x128 sauvegardée : {OUTPUT_PATH}")
