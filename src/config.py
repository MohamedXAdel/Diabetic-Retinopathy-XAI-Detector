import os
import pandas as pd
import torch

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

BASE_DIR = 'data/'  
NEW_DIR = 'data/Binary_DS'
SEV_DIR = 'data/Severity_DS'

TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'train.csv')
VALID_CSV_PATH = os.path.join(BASE_DIR, 'valid.csv')
TEST_CSV_PATH = os.path.join(BASE_DIR, 'test.csv')


TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train_images')
VAL_IMG_DIR = os.path.join(BASE_DIR, 'val_images')
TEST_IMG_DIR = os.path.join(BASE_DIR, 'test_images')

diagnosis_dict_binary = {0: 'No_DR', 1: 'DR', 2: 'DR', 3: 'DR', 4: 'DR'}
diagnosis_dict = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferate_DR'}


# Define source directory
train_src = "data/train_images"
val_src   = "data/val_images"
test_src  = "data/test_images"

# Load dataset
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)
df_val = pd.read_csv(VALID_CSV_PATH)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')