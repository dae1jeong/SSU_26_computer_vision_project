"""
데이터셋 로더: RESIDE (ITS, OTS, SOTS)
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class HazeDataset(Dataset):
    """
    RESIDE 데이터셋 로더
    구조:
        root/
            hazy/  ← 안개 낀 이미지
            clear/ ← 깨끗한 이미지 (GT)
    """
    def __init__(self, root, patch_size=256, is_train=True, augment=True):
        self.hazy_dir  = os.path.join(root, 'hazy')
        # RESIDE-6K는 'GT', 기존은 'clear'
        self.clear_dir = os.path.join(root, 'GT') if os.path.exists(os.path.join(root, 'GT')) else os.path.join(root, 'clear')
        self.patch_size = patch_size
        self.is_train   = is_train
        self.augment    = augment and is_train

        self.hazy_files = sorted(os.listdir(self.hazy_dir))

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        fname = self.hazy_files[idx]
        # RESIDE-6K: hazy/GT 파일명 동일 (1.jpg, 2.jpg ...)
        # RESIDE ITS: hazy={id}_{beta}_{A}.png → GT={id}.png
        clear_name = fname  # 기본: 같은 파일명
        if not os.path.exists(os.path.join(self.clear_dir, clear_name)):
            # ITS 형식: id_beta_A.ext → id.ext
            stem, ext = os.path.splitext(fname)
            base_id = stem.split('_')[0]
            for e in [ext, '.png', '.jpg']:
                candidate = base_id + e
                if os.path.exists(os.path.join(self.clear_dir, candidate)):
                    clear_name = candidate
                    break

        hazy  = Image.open(os.path.join(self.hazy_dir,  fname)).convert('RGB')
        clear = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        if self.is_train:
            hazy, clear = self._random_crop(hazy, clear)
            if self.augment:
                hazy, clear = self._augment(hazy, clear)
        else:
            hazy, clear = self._center_crop(hazy, clear)

        hazy  = TF.to_tensor(hazy)
        clear = TF.to_tensor(clear)
        return hazy, clear, fname

    def _random_crop(self, hazy, clear):
        i, j, h, w = T.RandomCrop.get_params(hazy, (self.patch_size, self.patch_size))
        return TF.crop(hazy, i, j, h, w), TF.crop(clear, i, j, h, w)

    def _center_crop(self, hazy, clear):
        W, H = hazy.size
        # 패치 크기 배수로 맞춤
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        hazy  = TF.center_crop(hazy,  [new_H, new_W])
        clear = TF.center_crop(clear, [new_H, new_W])
        return hazy, clear

    def _augment(self, hazy, clear):
        # Random horizontal flip
        if random.random() > 0.5:
            hazy, clear = TF.hflip(hazy), TF.hflip(clear)
        # Random vertical flip
        if random.random() > 0.5:
            hazy, clear = TF.vflip(hazy), TF.vflip(clear)
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            hazy  = TF.rotate(hazy,  k * 90)
            clear = TF.rotate(clear, k * 90)
        return hazy, clear


def get_dataloader(root, patch_size=256, batch_size=8, is_train=True,
                   num_workers=4, augment=True):
    dataset = HazeDataset(root, patch_size=patch_size, is_train=is_train, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
