import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

from Vocabulary import *


img_dir = 'D:/dataset/13.한국어글자체/01.손글씨/image'
label_file = 'D:/dataset/13.한국어글자체/01.손글씨/new_labels.csv'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

def split_dataset(label_file, test_size, shuffle):
    # Train, Test 데이터셋을 불러오는 함수
    labels = pd.read_csv(label_file)

    train_labels, test_labels = train_test_split(labels, test_size=test_size, shuffle=shuffle)

    return train_labels, test_labels

def collate_fn(batch, pad_idx):
    # pad_Sequence를 추가하는 함수

    images, input_labels, output_labels = zip(*batch)

    images = torch.stack(images, dim=0)

    input_labels_padded = pad_sequence(input_labels, batch_first=True, padding_value=pad_idx)
    output_labels_padded = pad_sequence(output_labels, batch_first=True, padding_value=pad_idx)

    return images, input_labels_padded, output_labels_padded


class HangulOCRDataset(Dataset):
    def __init__(self, img_dir, labels, transform = None):
        self.img_dir = img_dir
        self.labels = labels
        self.vocabulary = Vocabulary()
        self.transform = transform

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.labels.iloc[idx, 1]

        input_label, output_label = self.vocabulary.process(label)

        if self.transform:
            image = self.transform(image)

        return image, input_label, output_label

def return_one_batch():
    return images, input_labels, output_labels

train_labels, test_labels = split_dataset(label_file, test_size = 0.2, shuffle = True)

train_dataset = HangulOCRDataset(img_dir = img_dir, labels = train_labels, transform = transform)
test_dataset = HangulOCRDataset(img_dir = img_dir, labels = test_labels, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, collate_fn = lambda batch: collate_fn(batch, 53))
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True, collate_fn = lambda batch: collate_fn(batch, 53))

images, input_labels, output_labels = next(iter(train_loader))

