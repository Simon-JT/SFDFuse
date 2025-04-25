import os
import glob
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from unet import U2NETP
from torchvision import transforms

# Create a singleton ImagePreprocessor
_preprocessor_instance = None

def get_preprocessor(ir_dir="MSRS_train/ir", vis_dir="MSRS_train/vi"):
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = ImagePreprocessor(ir_dir, vis_dir)
    return _preprocessor_instance

class ImagePreprocessor:
    def __init__(self, ir_dir="MSRS_train/ir", vis_dir="MSRS_train/vi"):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        
        self.ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.*")))
        self.vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))

        self.u2net = None
        if torch.cuda.is_available():
            self._init_u2net()
        
    def _init_u2net(self):
        """Initialize U2NET model"""
        if self.u2net is None:
            print("Loading U2NET model...")
            self.u2net = U2NETP(in_ch=3, out_ch=1)
            
            weights_path = 'u2netp.pth'
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weight file not found: {weights_path}")
                
            state_dict = torch.load(weights_path, map_location="cuda")
            self.u2net.load_state_dict(state_dict)
            self.u2net.eval()
            self.u2net = self.u2net.cuda()

    def load_image_pair(self, idx):
        """Load image pair from IR and visible light directories"""
        ir_img = cv2.imread(self.ir_files[idx], cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(self.vis_files[idx], cv2.IMREAD_GRAYSCALE)

        ir_img = ir_img.astype(np.float32) / 255.0
        vis_img = vis_img.astype(np.float32) / 255.0

        ir_img = ir_img[None, :, :]
        vis_img = vis_img[None, :, :]
        
        return ir_img, vis_img

    def generate_mask(self, ir_img):
        """Generate binary mask from IR image"""

        # Take first dimension
        ir_np = ir_img[0]
        
        mask = np.zeros_like(ir_np)
        mask[ir_np > 0.1] = 1.0
        
        mask = mask[None, :, :]
        return mask

class FusionDataset(Dataset):
    def __init__(self, ir_dir, vis_dir):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.preprocessor = get_preprocessor(ir_dir, vis_dir)
        
        print("Preprocessing all images...")
        self.processed_data = []
        ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.*")))
        vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.*")))
        
        for idx in tqdm(range(len(ir_files))):
            ir_img, vis_img = self.preprocessor.load_image_pair(idx)
            mask = self.preprocessor.generate_mask(ir_img)
            
            # Convert to tensors
            ir_img = torch.tensor(ir_img, dtype=torch.float32)
            vis_img = torch.tensor(vis_img, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            
            self.processed_data.append((
                vis_img,
                ir_img,
                mask,
                (vis_img.shape[-2], vis_img.shape[-1])
            ))
        print("Preprocessing completed!")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        # Load and preprocess images
        ir_img, vis_img = self.preprocessor.load_image_pair(idx)
        
        # Generate mask
        mask = self.preprocessor.generate_mask(ir_img)

        size = (ir_img.shape[1], ir_img.shape[2])
        
        # Convert to torch tensors
        ir_img = torch.from_numpy(ir_img)
        vis_img = torch.from_numpy(vis_img)
        mask = torch.from_numpy(mask)
        
        return vis_img, ir_img, mask, size

def collate_fn(batch):
    vis_imgs, ir_imgs, masks, sizes = zip(*batch)
    return (
        torch.stack(vis_imgs),
        torch.stack(ir_imgs),
        torch.stack(masks),
        sizes
    )

def get_dataloader(ir_dir, vis_dir, batch_size=1, shuffle=False, num_workers=2):
    dataset = FusionDataset(ir_dir, vis_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True  
    )

if __name__ == "__main__":
    print("This script defines dataset preprocessing utilities.")



