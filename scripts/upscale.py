from super_image import EdsrModel, ImageLoader
from PIL import Image

# 1. Charger votre image restaurée en 128x128
image = Image.open('scripts\\resultat_restauration_128.png')

# 2. Charger le modèle pré-entraîné (ex: EDSR x4)
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)

# 3. Effectuer l'upscale (128x128 -> 512x512)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

# 4. Sauvegarder le résultat final "pro"
ImageLoader.save_image(preds, 'scripts\\output_final_512.png')
print("Upscale terminé !")