import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class CarConceptDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_loader(preprocess, n_samples, batch_size, num_workers=4, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    attribute_file = os.path.join("my_dataset", "data", "image_attribute_labels.txt")
    df = pd.read_csv(attribute_file, sep=" ", header=None)

    # Attribute sayısını otomatik al
    n_attributes = df.shape[1] - 2
    concept_loaders = {}

    for attr_idx in range(2, 2 + n_attributes):  # Attribute kolonları
        concept_name = f"attr_{attr_idx - 2}"
        print(f"Preparing concept: {concept_name}")
        pos_df = df[df[attr_idx] == 1]
        neg_df = df[df[attr_idx] == 0]

        if len(pos_df) < 2 * n_samples:
            pos_df = pos_df.sample(2 * n_samples, replace=True, random_state=seed)
        else:
            pos_df = pos_df.sample(2 * n_samples, random_state=seed)

        if len(neg_df) < 2 * n_samples:
            neg_df = neg_df.sample(2 * n_samples, replace=True, random_state=seed)
        else:
            neg_df = neg_df.sample(2 * n_samples, random_state=seed)

        pos_paths = [os.path.join("my_dataset", "dataset", row[1]) for _, row in pos_df.iterrows()]
        neg_paths = [os.path.join("my_dataset", "dataset", row[1]) for _, row in neg_df.iterrows()]

        pos_dataset = CarConceptDataset(pos_paths, transform=preprocess)
        neg_dataset = CarConceptDataset(neg_paths, transform=preprocess)

        pos_loader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        neg_loader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        concept_loaders[concept_name] = {"pos": pos_loader, "neg": neg_loader}

    return concept_loaders


def load_my_dataset(preprocess=None):
    data_dir = "my_dataset/dataset"

    transform = preprocess if preprocess else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Eğer preprocess zaten ToTensor() içeriyorsa, burayı kaldır
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    classes = train_dataset.classes
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    return train_loader, test_loader, idx_to_class, classes