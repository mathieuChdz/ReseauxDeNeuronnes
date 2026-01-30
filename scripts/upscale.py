import os
from PIL import Image
from tqdm import tqdm

RESTORED_DIR = "./restored_outputs_train"
HR_DIR = "../data/train/train2014"
OUT_DIR = "restored_upscaled_hr"

N = 100  # seulement les 100 premières

os.makedirs(OUT_DIR, exist_ok=True)

restored_files = sorted(os.listdir(RESTORED_DIR))[:N]
hr_files = sorted(os.listdir(HR_DIR))[:N]

for i in tqdm(range(N), desc="Upscaling restored images"):
    rest_img = Image.open(
        os.path.join(RESTORED_DIR, restored_files[i])
    ).convert("RGB")

    hr_img = Image.open(
        os.path.join(HR_DIR, hr_files[i])
    ).convert("RGB")

    #  resize vers la taille HR
    rest_up = rest_img.resize(hr_img.size, Image.BICUBIC)

    # nom propre basé sur la HR
    out_name = hr_files[i]
    rest_up.save(os.path.join(OUT_DIR, out_name), quality=95)
print(" Upscaling done. Upscaled images saved in:", OUT_DIR)