import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class StippleDataset(Dataset):
    """Dataset that yields (condition, offset) pairs for ControlNet training.

    Parameters
    ----------
    condition_path : str
        Directory of grayscale source images (any common format).
    offset_path : str
        Directory of pre-computed offset ``.npy`` files produced by
        ``prepare_data.py``.  Filenames must correspond to the condition
        images (same stem, ``.npy`` extension).
    grid_size : int
        Spatial resolution of the offset grid (default 32 -> 32x32 = 1024
        points).
    """

    def __init__(self, condition_path, offset_path, grid_size=32):
        self.condition_path = condition_path
        self.offset_path = offset_path
        self.grid_size = grid_size

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        self.filenames = sorted(
            f
            for f in os.listdir(condition_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        # --- Condition: grayscale image resized to grid_size x grid_size ---
        img = cv2.imread(
            os.path.join(self.condition_path, filename), cv2.IMREAD_GRAYSCALE
        )
        img = cv2.resize(
            img, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA
        )
        cond = torch.from_numpy(img).float() / 255.0
        cond = cond.unsqueeze(0)  # (1, H, W)

        # --- Target: offset tensor of shape (2, grid_size, grid_size) ---
        offsets = np.load(os.path.join(self.offset_path, stem + ".npy"))
        offsets = torch.from_numpy(offsets).float()  # already (2, H, W)

        return cond, offsets
