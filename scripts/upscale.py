from PIL import Image
import torch
from py_real_esrgan.model import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modèle x4
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# ton image restaurée 128x128
img = Image.open('assets\\before.jpg').convert('RGB')

# upscale
sr = model.predict(img)  # ~512x512
sr.save('assets\\after.jpg')
