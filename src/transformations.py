import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from src.preprocess import BinaryRetinopathyDataset,SeverityDataset,preprocess_image
from src.config import *


# Data Transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

train_dataset = BinaryRetinopathyDataset(
    root_dir=NEW_DIR,
    split='train',
    transform=train_transform
)

val_dataset = BinaryRetinopathyDataset(
    root_dir=NEW_DIR,
    split='val',
    transform=val_transform
)


test_dataset = BinaryRetinopathyDataset(
    root_dir=NEW_DIR,
    split='test',
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

#-------------------------------------------------------------------------------------------------------------------------

# Multi class transformations
sev_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sev_val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sev_train_ds = SeverityDataset(SEV_DIR, split='train', transform=sev_train_transform)
sev_val_ds   = SeverityDataset(SEV_DIR, split='val',   transform=sev_val_transform)
sev_test_ds  = SeverityDataset(SEV_DIR, split='test',  transform=sev_val_transform)

# Multi data loader
sev_train_loader = DataLoader(sev_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
sev_val_loader   = DataLoader(sev_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
sev_test_loader  = DataLoader(sev_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)



#Calculate Class Weights for Imbalance
# We use sklearn's class weight to punish the model more for missing rare classes like class 3
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(sev_train_ds.labels), 
    y=sev_train_ds.labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


#-------------------------------------------------------------------------------------------------------------------------

# Define Final Test Dataset (Original 0-4 Labels)
class FinalTestDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Images are scattered in subfolders in our 'Binary_DS' or 'Severity_DS'
        # But it's easier to read from the original source folders if we have the paths.
        # However, we moved them. Let's look for them in the 'test_images' source folder directly 
        # to be safe, OR check where we put them. 
        
        # We know all test images are in /kaggle/input/aptos2019/test_images/test_images
        # Let's use the original source path to be absolutely sure we find them.
        fname = row['id_code'] + '.png'
        img_path = os.path.join("/kaggle/input/aptos2019/test_images/test_images", fname)
        
        # Preprocess
        image = preprocess_image(img_path, sigmaX=10)
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and Original Diagnosis (0-4)
        return image, row['diagnosis']


# 2. Create the Final Loader 
# We use df_test which still has the 'diagnosis' column (0-4)
final_test_ds = FinalTestDataset(df_test, TEST_IMG_DIR, transform=val_transform)
final_test_loader = DataLoader(final_test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# 3. Combined Prediction Function 
def get_two_stage_prediction(binary_model, severity_model, image_tensor, threshold=0.5):
    # Ensure models are in eval mode
    binary_model.eval()
    severity_model.eval()
    
    with torch.no_grad():
        #Stage 1: Binary
        bin_logits = binary_model(image_tensor)
        bin_prob = torch.sigmoid(bin_logits).item()
        
        if bin_prob < threshold:
            return 0  # No DR
        
        # Stage 2: Severity
        # If we are here, the model thinks it IS DR.
        # Pass the SAME image to the severity model.
        sev_logits = severity_model(image_tensor)
        _, sev_pred_idx = torch.max(sev_logits, 1) # Returns 0, 1, 2, or 3
        
        final_pred = sev_pred_idx.item() + 1 # Convert back to 1, 2, 3, 4
        
        return final_pred

