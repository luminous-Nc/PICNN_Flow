from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import os


# Custom Pytorch Dataset
class LBMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, 'sdf'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, 'sdf', img_name)
        label_name = img_name.split('_')[0]  # Get number
        label_x_path = os.path.join(self.root_dir, 'lbm', f"{label_name}_lbmx.png")
        label_y_path = os.path.join(self.root_dir, 'lbm', f"{label_name}_lbmy.png")

        img = Image.open(img_path).convert('RGB')
        label_x = Image.open(label_x_path).convert('L')
        label_y = Image.open(label_y_path).convert('L')

        if self.transform:
            img = self.transform(img)
            label_x = self.transform(label_x)
            label_y = self.transform(label_y)

        return img, label_x, label_y
