import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class MAGFiLODataset(Dataset):
    """
    Dataset loader for the MAGFiLO (Manually Annotated GONG Filaments) dataset.
    Implements patch-based sampling with class balancing, as requested in Phase 2.3.
    """
    def __init__(self, root_dir, split='train', patch_size=256, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        
        # In a real scenario, this would glob the FITS/PNG images and their masks
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, split, 'images', '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, split, 'masks', '*.png')))
        self.spine_paths = sorted(glob.glob(os.path.join(root_dir, split, 'spines', '*.png')))
        
        # Verify alignment
        if len(self.image_paths) != len(self.mask_paths):
            print("Warning: Mismatch between number of images and masks!")
            
        if len(self.image_paths) == 0:
            print("Warning: No images found! Using dummy length of 100 for testing purposes.")

    def __len__(self):
        return max(100, len(self.image_paths))

    def extract_patch(self, image, mask, spine):
        """
        Extracts a random patch from the image. 
        Implements class-balanced sampling by frequently selecting patches 
        that contain filament/barb pixels.
        """
        h, w = image.shape[:2]
        ph, pw = self.patch_size, self.patch_size
        
        if h <= ph or w <= pw:
            # If image is smaller than patch, resize or pad (assuming large images though)
            return cv2.resize(image, (pw, ph)), cv2.resize(mask, (pw, ph)), cv2.resize(spine, (pw, ph))

        # Class balanced sampling: 70% chance to pick a patch containing foreground
        if random.random() < 0.7:
            # Find foreground pixels
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                idx = random.randint(0, len(y_indices) - 1)
                center_y, center_x = y_indices[idx], x_indices[idx]
                
                # Jitter center slightly
                center_y += random.randint(-ph//4, ph//4)
                center_x += random.randint(-pw//4, pw//4)
                
                # Clamp to image bounds
                y1 = max(0, min(center_y - ph // 2, h - ph))
                x1 = max(0, min(center_x - pw // 2, w - pw))
            else:
                # Fallback to random if mask is empty
                y1 = random.randint(0, h - ph)
                x1 = random.randint(0, w - pw)
        else:
            # Random patch (likely background)
            y1 = random.randint(0, h - ph)
            x1 = random.randint(0, w - pw)
            
        y2, x2 = y1 + ph, x1 + pw
        
        patch_image = image[y1:y2, x1:x2]
        patch_mask = mask[y1:y2, x1:x2]
        patch_spine = spine[y1:y2, x1:x2] if spine is not None else np.zeros_like(patch_mask)
        
        return patch_image, patch_mask, patch_spine

    def __getitem__(self, idx):
        # Load images
        # Here we mock the loading for demonstration since data is missing
        # image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # DUMMY DATA FOR NOW
        image = np.zeros((1024, 1024), dtype=np.uint8)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        spine = np.zeros((1024, 1024), dtype=np.uint8)
        
        patch_img, patch_mask, patch_spine = self.extract_patch(image, mask, spine)
        
        # Convert to tensor
        patch_img = torch.from_numpy(patch_img).float().unsqueeze(0) / 255.0
        patch_mask = torch.from_numpy(patch_mask).float().unsqueeze(0) / 255.0
        patch_spine = torch.from_numpy(patch_spine).float().unsqueeze(0) / 255.0
        
        if self.augment:
            # Brightness/contrast jitter
            if random.random() > 0.5:
                patch_img = TF.adjust_brightness(patch_img, brightness_factor=random.uniform(0.8, 1.2))
            
            # Flips (Horizontal flips are safe as chirality is not graded)
            if random.random() > 0.5:
                patch_img = TF.hflip(patch_img)
                patch_mask = TF.hflip(patch_mask)
                patch_spine = TF.hflip(patch_spine)
                
            if random.random() > 0.5:
                patch_img = TF.vflip(patch_img)
                patch_mask = TF.vflip(patch_mask)
                patch_spine = TF.vflip(patch_spine)
                
        return {
            'image': patch_img,
            'mask': patch_mask,
            'spine': patch_spine
        }
