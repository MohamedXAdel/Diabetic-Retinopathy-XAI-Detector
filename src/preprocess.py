import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from src.config import IMG_SIZE

# Ben Graham's Preprocessing Method 
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    

def preprocess_image(image_path, sigmaX=10):
    """
    Load → BGR2RGB → crop gray borders → resize to IMG_SIZE → Ben Graham contrast
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        return Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)               # remove gray borders
    
    # Resize to the global IMG_SIZE (preserves aspect ratio only if you want it;
    # here we force square like the original code)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Ben Graham's contrast enhancement
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    image = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    # Ensure values stay in [0, 255] and uint8 (important for PIL)
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image)



# Custom Binary Dataset Class
class BinaryRetinopathyDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        if split == 'test':
            for label in ['0', '1']:  # iterate over subfolders
                folder = os.path.join(root_dir, 'test', label)
                if not os.path.exists(folder):
                    continue  # skip if a subfolder is missing
                for fname in os.listdir(folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(folder, fname))
                        self.labels.append(int(label))
        else:
            for label in ['0', '1']:
                folder = os.path.join(root_dir, split, label)
                if not os.path.exists(folder):
                    raise FileNotFoundError(f"Folder not found: {folder}")
                for fname in os.listdir(folder):
                    if fname.endswith('.png'):
                        self.images.append(os.path.join(folder, fname))
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = preprocess_image(img_path, sigmaX=10)
        
        if self.transform:
            image = self.transform(image)

        if label == -1:  # test set
            return image, os.path.basename(img_path)  # return filename for submission
        return image, torch.tensor(label, dtype=torch.long)
    

#-------------------------------------------------------------------------------------------------------------------------

#Severity Dataset Class
class SeverityDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Mapping: 1->0, 2->1, 3->2, 4->3
        self.label_map = {'1': 0, '2': 1, '3': 2, '4': 3}
        # Reverse mapping for later use if needed
        self.reverse_map = {0: '1', 1: '2', 2: '3', 3: '4'}

        base_path = os.path.join(root_dir, split)
        # Iterate over folders '1', '2', '3', '4'
        for label_name in ['1', '2', '3', '4']:
            class_dir = os.path.join(base_path, label_name)
            if not os.path.exists(class_dir):
                continue
                
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg')):
                    self.images.append(os.path.join(class_dir, fname))
                    # Convert original label '1' to 0, etc.
                    self.labels.append(self.label_map[label_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Reuse the preprocessing function from your previous code
        image = preprocess_image(img_path, sigmaX=10)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    

    