import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# 🔹 Concept Names
concept_names = [
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large",
    "compact", "boxy", "sloped_rear", "roof_rails", "offroad", "flat_front", "luxury"
]

# 🔹 Dataset
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
    def __init__(self, num_concepts=20):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_concepts)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# 🔹 Label Predictor
class LabelPredictor(nn.Module):
    def __init__(self, input_dim=20, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 🔹 Load Data
column_names = ["label", "image_path"] + concept_names
df = pd.read_csv("my_dataset/data/image_attribute_labels.txt", sep="\s+", header=None, names=column_names)
dataset = CarConceptDataset(df, root_dir="my_dataset/dataset")

# 🔹 Load Trained Models
concept_model = ConceptPredictor()
concept_model.load_state_dict(torch.load("concept_model.pth", map_location="cpu"))
concept_model.eval()

label_model = LabelPredictor()
label_model.load_state_dict(torch.load("label_model.pth", map_location="cpu"))
label_model.eval()

# 🔁 Inference with Explanation
with torch.no_grad():
    for i in range(len(dataset)):
        image, _, true_label, image_path = dataset[i]
        image = image.unsqueeze(0)

        predicted_concepts = concept_model(image)
        binary_list = predicted_concepts.squeeze(0).round().int().tolist()
        active_concepts = [name for i, name in enumerate(concept_names) if binary_list[i] == 1]

        predicted_class_logits = label_model(predicted_concepts)
        predicted_class = predicted_class_logits.argmax(dim=1).item()

        print(f"\n📷 {image_path}")
        print(f"🔢 Tahmin edilen conceptler: {active_concepts}")
        print(f"🎯 Tahmin edilen sınıf: {dataset.idx_to_label[predicted_class]}")
