import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DiffSpotDataset(Dataset):
    """
    Diff task Dataset for data/diff/spot-the-diff-harsh19/ data structure.
    Each sample outputs:
      - images: [2, 3, H, W] (before/after)
      - diff_map: [H, W] (grayscale difference map)
      - labels: text description list
      - task_id: int (all 0)
      - img_ids, rows, cols: token metadata
    """
    def __init__(self, ann_path, img_dir, prompt_len=5, patch_size=16, num_patches_per_row=14, num_patches_per_col=14, transform=None, label_tokenizer=None, use_augmentation=False):
        super().__init__()
        self.ann = self._load_json(ann_path)
        self.img_dir = img_dir
        self.prompt_len = prompt_len
        self.patch_size = patch_size
        self.num_patches_per_row = num_patches_per_row
        self.num_patches_per_col = num_patches_per_col
        self.num_patches = num_patches_per_row * num_patches_per_col
        self.transform = transform
        self.label_tokenizer = label_tokenizer  # Optional: text to token conversion
        self.use_augmentation = use_augmentation
        
        # Data augmentation transforms
        if use_augmentation:
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.RandomRotation(degrees=5),
            ])
        else:
            self.aug_transform = None

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        sample = self.ann[idx]
        img_id = sample['img_id']
        
        # Load image pair
        img1_path = os.path.join(self.img_dir, f'{img_id}.png')
        img2_path = os.path.join(self.img_dir, f'{img_id}_2.png')
        diff_path = os.path.join(self.img_dir, f'{img_id}_diff.jpg')
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        diff_map = Image.open(diff_path).convert('L')
        
        base_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        img1_tensor = base_transform(img1)
        img2_tensor = base_transform(img2)
        
        # [Key Fix] Correctly process diff_map
        # diff_map should be processed as a difference map, not a regular image
        diff_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        diff_map_tensor = diff_transform(diff_map)
        
        # [IMPROVED] Better diff_map processing
        # Always calculate real difference between image pair for more accurate results
        real_diff = torch.abs(img1_tensor - img2_tensor).mean(dim=0, keepdim=True)
        
        # Use the original diff_map if it's meaningful, otherwise use calculated difference
        if diff_map_tensor.mean() < 0.5:  # If original diff_map has low values, it might be meaningful
            # Combine original and calculated difference
            diff_map_tensor = torch.maximum(diff_map_tensor, real_diff)
        else:
            # Use calculated difference
            diff_map_tensor = real_diff
        
        # Normalize to [0, 1] range
        if diff_map_tensor.max() > 0:
            diff_map_tensor = diff_map_tensor / diff_map_tensor.max()
        
        # Add some noise to avoid all zeros
        diff_map_tensor = torch.clamp(diff_map_tensor + torch.rand_like(diff_map_tensor) * 0.01, 0, 1)
        
        if self.use_augmentation and self.aug_transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img1_tensor = self.aug_transform(img1_tensor)
            torch.manual_seed(seed)
            img2_tensor = self.aug_transform(img2_tensor)
            
        images = torch.stack([img1_tensor, img2_tensor], dim=0)
        
        diff_map = diff_map_tensor.squeeze(0)
        
        labels = sample['sentences']
        if self.label_tokenizer:
            labels = self.label_tokenizer(labels)
        else:
            if isinstance(labels, list) and len(labels) > 0:
                labels = labels[0]
            else:
                labels = "no differences"
        
        total_tokens = self.prompt_len + 2 * self.num_patches
        img_ids_meta = [-1]*self.prompt_len + [0]*self.num_patches + [1]*self.num_patches
        rows = [-1]*self.prompt_len + [i//self.num_patches_per_row for i in range(self.num_patches)]*2
        cols = [-1]*self.prompt_len + [i%self.num_patches_per_col for i in range(self.num_patches)]*2
        
        return {
            'images': images,
            'diff_map': diff_map,
            'labels': labels,
            'task_id': 0,
            'img_ids': torch.tensor(img_ids_meta, dtype=torch.long),
            'rows': torch.tensor(rows, dtype=torch.long),
            'cols': torch.tensor(cols, dtype=torch.long),
            'img1_path': img1_path,
            'img2_path': img2_path,
        } 