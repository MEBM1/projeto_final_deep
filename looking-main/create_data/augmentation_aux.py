import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class JAADDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data  # Dicionário com as anotações (como o gerado pela `generate_with_attributes_ann`)
        self.root_dir = root_dir  # Caminho base para as imagens
        self.transform = transform  # Transformações a serem aplicadas

    def __len__(self):
        return len(self.data["path"])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data["path"][idx])
        image = Image.open(img_path).convert('RGB')  # Abre a imagem em RGB
        
        if self.transform:
            image = self.transform(image)  # Aplica as transformações
        
        label = torch.tensor(self.data["Y"][idx], dtype=torch.float32)  # Converte o rótulo para tensor
        return image, label
