import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# 🔹 Dataset (Aynı)
class CarConceptDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.root_dir}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        concept_labels = torch.tensor(row[2:].astype(float).values, dtype=torch.float32)
        class_label = torch.tensor(self.label_to_idx[row['label']], dtype=torch.long)

        return image, concept_labels, class_label, row['image_path']

# 🔹 Concept Predictor
class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts=13):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_concepts)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# 🔹 Label Predictor
class LabelPredictor(nn.Module):
    def __init__(self, input_dim=13, num_classes=9):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 🔹 Veriyi oku
column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large"
]

df = pd.read_csv(
    "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt",
    sep="\s+", header=None, names=column_names
)

dataset = CarConceptDataset(df, root_dir="/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/dataset")

# 🔹 Model nesneleri (yüklenmiş haliyle test)
concept_model = ConceptPredictor()
label_model = LabelPredictor()

concept_model.eval()
label_model.eval()

# 🔁 İlk örneği alalım
image, true_concepts, true_label, image_path = dataset[0]
image = image.unsqueeze(0)

# 🔄 CBM forward: Görsel → Kavram → Sınıf
with torch.no_grad():
    predicted_concepts = concept_model(image)
    predicted_class_logits = label_model(predicted_concepts)
    predicted_class_idx = predicted_class_logits.argmax(dim=1).item()

# 🔎 Çıktılar
idx_to_label = dataset.idx_to_label
print(f"\n🖼️ Görsel: {image_path}")
print(f"🎯 Tahmin edilen kavramlar: {predicted_concepts.squeeze().round(decimals=2)}")
print(f"✅ Gerçek sınıf       : {idx_to_label[true_label.item()]}")
print(f"🤖 Tahmin edilen sınıf: {idx_to_label[predicted_class_idx]}")
