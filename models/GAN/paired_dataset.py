import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, root):
        self.target_dir = os.path.join(root, "images")
        self.deg_dir = os.path.join(root, "degraded_images")

        self.target_names = sorted(os.listdir(self.target_dir))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.target_names)

    def __getitem__(self, idx):
        target_name = self.target_names[idx]

        # --- construire le nom de l'image dégradée ---
        # image_00183.jpg -> degraded_image_00183.jpg
        deg_name = "degraded_" + target_name

        target_path = os.path.join(self.target_dir, target_name)
        deg_path = os.path.join(self.deg_dir, deg_name)

        if not os.path.exists(deg_path):
            raise FileNotFoundError(deg_path)

        target = Image.open(target_path).convert("RGB")
        degraded = Image.open(deg_path).convert("RGB")

        target = self.to_tensor(target)
        degraded = self.to_tensor(degraded)

        return degraded, target
