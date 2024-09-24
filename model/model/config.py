import torch
from torchvision import datasets, transforms

PAD_TOKEN = 53
CHECKPOINT_DIR = "./checkpoint"

BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 1

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
ADAM_EPS = 5e-9
SCHEDULER_FACTOR = 0.9
SCHEDULER_PATIENCE = 10

DROPOUT_RATE = 0.1

IMG_DIR = 'D:/dataset/13.한국어글자체/01.손글씨/image'
LABEL_FILE = 'D:/dataset/13.한국어글자체/01.손글씨/new_labels.csv'

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
