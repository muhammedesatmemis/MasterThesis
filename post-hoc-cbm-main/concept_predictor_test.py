import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

# 🔹 Dataset Sınıfı
class CarConceptDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.root_dir}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        concept_labels = torch.tensor(row[2:].astype(float).values, dtype=torch.float32)
        class_label = torch.tensor(self.label_to_idx[row['label']], dtype=torch.long)

        return image, concept_labels, class_label

# 🔹 Concept Predictor Modeli
class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts=13):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_concepts)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))  # Multi-label için sigmoid

# 🔹 CSV Yükleme
column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large"
]

df = pd.read_csv(
    "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt",
    sep="\s+", header=None, names=column_names
)

# 🔹 Dataset ve Model
dataset = CarConceptDataset(df, root_dir="/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/dataset")
model = ConceptPredictor()

# 🔹 1 Örnekle Test
image, concept_labels, _ = dataset[0]
image = image.unsqueeze(0)  # batch dimension

with torch.no_grad():
    predicted_concepts = model(image)

print("🎯 Tahmin Edilen Kavramlar:", predicted_concepts.squeeze().round(decimals=3))
print("✅ Gerçek Kavramlar       :", concept_labels)
