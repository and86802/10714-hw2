from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        
        with gzip.open(label_filename, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        images = images.reshape((-1, 28, 28, 1))
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        img = self.apply_transforms(img)
        return img.reshape(-1, 28*28), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION