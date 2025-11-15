import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from PIL import Image


class PixelDataset(Dataset):
    """
    Pytorch dataset for sampling pixels from a single image.

    This will flatten an image into a list of 2D coordinates and RGB values. The coordinates are normalized based on the image width/height.
    Specifically it will save 2D coordinates of shape (H*W, 2) and colors of shape (H*W, 3) so we can randomly sample N values later.
    """

    def __init__(self, image_path):
        super().__init__()
        
        try:
            img = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")

        img = np.array(img)
        self.height, self.width = img.shape[:2]

        img_tensor_normalized = torch.from_numpy(img).float() / 255.0

        # Saving colors
        self.pixels = torch.reshape(img_tensor_normalized, (-1, 3))

        x_coords = torch.arange(self.width, dtype=torch.float32)
        y_coords = torch.arange(self.height, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        xx_normalized = xx / self.width
        yy_normalized = yy / self.height
        coords = torch.stack((xx_normalized, yy_normalized), dim=-1)

        # Saving the coordinates (H*W, 2)
        self.coords = torch.reshape(coords, (-1, 2))

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]
    
