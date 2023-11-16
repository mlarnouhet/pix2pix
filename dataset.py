import os
from torch.utils.data import Dataset
import numpy as np
from fastai.vision.all import show_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
from torch.utils.data import DataLoader
from tqdm import tqdm
matplotlib.use('Qt5Agg')

class cityscape(Dataset):
    def __init__(self, data_dir, mode, transforms):
        self.image_dir = data_dir+ "\\"+mode+ "\\image"
        self.mask_dir = data_dir+ "\\"+mode+ "\\label"
        self.mode = mode
        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(np.load(os.path.join(self.image_dir,image_path)))
        mask = np.array(np.load(os.path.join(self.mask_dir, mask_path)))

        if self.transforms is not None:
            augmentation = self.transforms(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask



def test():
    transforms = A.Compose(
[
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
],
)
    dataset = cityscape("data", "train", transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_id, (image, mask) in enumerate(tqdm(dataloader)):
        print(image.shape)

if __name__ == "__main__":
    test()
    print("end")