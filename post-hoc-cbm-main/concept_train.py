import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

# 🔹 Attribute isimleri
column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large",
    "compact", "boxy", "sloped_rear", "roof_rails", "offroad", "flat_front", "luxury"
]

# 🔹 Dataset sınıfı
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

# 🔹 Model
class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts=20):  # 🔸 20 concept var
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_concepts)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))  # Multi-label classification için sigmoid

# 🔹 Veri yükle
df = pd.read_csv(
    "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt",
    sep="\s+", header=None, names=column_names
)

dataset = CarConceptDataset(df, root_dir="/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/dataset")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🔹 Eğitim ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConceptPredictor(num_concepts=20).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 🔁 Eğitim döngüsü
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, concepts, _ in dataloader:
        images, concepts = images.to(device), concepts.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, concepts)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📚 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "concept_model.pth")
print("✅ Eğitim tamamlandı.")
